using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using Xunit;

namespace ANN.Net.Tests.Embeddings
{
    public class Word2VecNetworkTests
    {
        private const int CLUSTER_SIZE = 4;
        private const int VOCABULARY_SIZE = 2 * CLUSTER_SIZE;

        private static Word2VecSettings CreateSettings(Word2VecArchitecture architecture = Word2VecArchitecture.SkipGram, int seed = 7)
        {
            return new Word2VecSettings
            {
                VocabularySize = VOCABULARY_SIZE,
                EmbeddingDimension = 8,
                Architecture = architecture,
                WindowSize = 2,
                UseDynamicWindow = false,
                NegativeSamplesCount = 3,
                SubsamplingThreshold = 0,
                Seed = seed,
                OptimizerType = OptimizerTypes.Normal
            };
        }

        private static IEmbeddingNetwork CreateNetwork(Word2VecSettings settings, float learningRate)
        {
            IEmbeddingNetwork network = NetworkFactory.BuildWord2Vec(settings);
            foreach (LearningRateOptimizer optimizer in network.Optimizers)
            {
                optimizer.LearningRate = learningRate;
            }

            return network;
        }

        /// <summary>
        /// Two clusters of tokens; every sequence stays inside one cluster, so tokens of a cluster
        /// only ever see each other as context
        /// </summary>
        private static List<int[]> ClusterCorpus(int sequencesPerCluster = 60, int sequenceLength = 8, int seed = 11)
        {
            Random random = new Random(seed);
            List<int[]> corpus = new List<int[]>();
            for (int cluster = 0; cluster < 2; cluster++)
            {
                for (int s = 0; s < sequencesPerCluster; s++)
                {
                    int[] sequence = new int[sequenceLength];
                    for (int t = 0; t < sequenceLength; t++)
                    {
                        sequence[t] = cluster * CLUSTER_SIZE + random.Next(CLUSTER_SIZE);
                    }

                    corpus.Add(sequence);
                }
            }

            return corpus;
        }

        private static int ClusterOf(int token)
        {
            return token / CLUSTER_SIZE;
        }

        private static double MeanSimilarity(IEmbeddingNetwork network, bool sameCluster)
        {
            double sum = 0;
            int count = 0;
            for (int first = 0; first < VOCABULARY_SIZE; first++)
            {
                for (int second = first + 1; second < VOCABULARY_SIZE; second++)
                {
                    if ((ClusterOf(first) == ClusterOf(second)) != sameCluster) continue;

                    sum += network.CosineSimilarity(first, second);
                    count++;
                }
            }

            return sum / count;
        }

        [Fact]
        public void Propagate_ReturnsOneScorePerOutputToken()
        {
            IEmbeddingNetwork network = CreateNetwork(CreateSettings(), 0);

            Quad[] scores = network.Propagate(new[] { 1 }, new[] { 2, 3, 4 });

            Assert.Equal(3, scores.Length);
            Assert.All(scores, score => Assert.True(score > 0 && score < 1));
            Assert.Same(scores, network.Values);
        }

        [Fact]
        public void Propagate_FreshNetwork_ScoresAreOneHalf()
        {
            IEmbeddingNetwork network = CreateNetwork(CreateSettings(), 0);

            // output embeddings start at zero, so every logit is zero
            Quad[] scores = network.Propagate(new[] { 0, 1 }, new[] { 2 });

            Assert.Equal(0.5f, (float)scores[0], 5);
        }

        [Fact]
        public void Backpropagate_SameSampleRepeatedly_LossDecreases()
        {
            IEmbeddingNetwork network = CreateNetwork(CreateSettings(), 0.1f);
            int[] inputs = { 0 };
            int[] outputs = { 1, 5, 6 };
            Quad[] targets = { 1, 0, 0 };

            network.Propagate(inputs, outputs);
            Quad firstLoss = network.Backpropagate(targets);
            Assert.Equal(3, network.Errors.Length);

            // the output vectors start at zero, so the first steps are slow
            Quad lastLoss = firstLoss;
            for (int i = 0; i < 500; i++)
            {
                network.Propagate(inputs, outputs);
                lastLoss = network.Backpropagate(targets);
            }

            Assert.True(lastLoss < firstLoss, $"Loss went from {firstLoss} to {lastLoss}");
            Assert.True(network.Values[0] > 0.9f, $"Context score {network.Values[0]} is too low");
            Assert.True(network.Values[1] < 0.1f, $"Noise score {network.Values[1]} is too high");
        }

        [Fact]
        public void Backpropagate_WithZeroLearningRate_DoesNotChangeEmbeddings()
        {
            IEmbeddingNetwork network = CreateNetwork(CreateSettings(), 0);
            Quad[] before = network.GetEmbedding(0);

            network.Propagate(new[] { 0 }, new[] { 1, 2 });
            network.Backpropagate(new Quad[] { 1, 0 });

            Assert.Equal(before.Select(v => (float)v), network.GetEmbedding(0).Select(v => (float)v));
        }

        [Theory]
        [InlineData(Word2VecArchitecture.SkipGram)]
        [InlineData(Word2VecArchitecture.ContinuousBagOfWords)]
        public void Train_OnClusteredCorpus_SameClusterTokensAreCloser(Word2VecArchitecture architecture)
        {
            IEmbeddingNetwork network = CreateNetwork(CreateSettings(architecture), 0.05f);
            List<int[]> corpus = ClusterCorpus();

            for (int epoch = 0; epoch < 15; epoch++)
            {
                network.Train(corpus);
            }

            double within = MeanSimilarity(network, sameCluster: true);
            double across = MeanSimilarity(network, sameCluster: false);

            Assert.True(within > across, $"Within cluster {within} should exceed across clusters {across}");
            Assert.True(within > 0.5, $"Within cluster similarity {within} is too low");
            Assert.True(across < 0.25, $"Across clusters similarity {across} is too high");
        }

        [Fact]
        public void MostSimilar_AfterTraining_ReturnsClusterMates()
        {
            IEmbeddingNetwork network = CreateNetwork(CreateSettings(), 0.05f);
            List<int[]> corpus = ClusterCorpus();

            for (int epoch = 0; epoch < 15; epoch++)
            {
                network.Train(corpus);
            }

            IReadOnlyList<(int Token, Quad Similarity)> similar = network.MostSimilar(5, CLUSTER_SIZE - 1);

            Assert.Equal(CLUSTER_SIZE - 1, similar.Count);
            Assert.All(similar, s => Assert.Equal(1, ClusterOf(s.Token)));
            Assert.DoesNotContain(similar, s => s.Token == 5);
            for (int i = 1; i < similar.Count; i++)
            {
                Assert.True(similar[i - 1].Similarity >= similar[i].Similarity);
            }
        }

        [Fact]
        public void Train_LossDecreasesOverEpochs()
        {
            IEmbeddingNetwork network = CreateNetwork(CreateSettings(), 0.05f);
            List<int[]> corpus = ClusterCorpus();

            Quad firstLoss = network.Train(corpus);
            Quad lastLoss = firstLoss;
            for (int epoch = 0; epoch < 10; epoch++)
            {
                lastLoss = network.Train(corpus);
            }

            Assert.True(firstLoss > 0);
            Assert.True(lastLoss < firstLoss, $"Loss went from {firstLoss} to {lastLoss}");
        }

        [Fact]
        public void Train_WithSameSeed_IsReproducible()
        {
            List<int[]> corpus = ClusterCorpus();

            IEmbeddingNetwork first = CreateNetwork(CreateSettings(seed: 3), 0.05f);
            IEmbeddingNetwork second = CreateNetwork(CreateSettings(seed: 3), 0.05f);

            Quad firstLoss = first.Train(corpus);
            Quad secondLoss = second.Train(corpus);

            Assert.Equal((float)firstLoss, (float)secondLoss);
            Assert.Equal(first.GetEmbedding(2).Select(v => (float)v), second.GetEmbedding(2).Select(v => (float)v));
        }

        [Fact]
        public void Train_WithSubsamplingAndDynamicWindow_Runs()
        {
            Word2VecSettings settings = CreateSettings();
            settings.SubsamplingThreshold = 0.05f;
            settings.UseDynamicWindow = true;
            IEmbeddingNetwork network = CreateNetwork(settings, 0.05f);

            Quad loss = network.Train(ClusterCorpus());

            Assert.True(loss > 0);
            Assert.False(float.IsNaN(loss));
        }

        [Fact]
        public void Train_EmptyCorpus_ReturnsZero()
        {
            IEmbeddingNetwork network = CreateNetwork(CreateSettings(), 0.05f);

            Assert.Equal(0f, (float)network.Train(new List<int[]>()));
            Assert.Equal(0f, (float)network.Train(new[] { Array.Empty<int>() }));
        }

        [Fact]
        public void GetEmbedding_ReturnsCopyOfTheRightSize()
        {
            IEmbeddingNetwork network = CreateNetwork(CreateSettings(), 0);

            Quad[] embedding = network.GetEmbedding(3);
            embedding[0] = 42;

            Assert.Equal(8, embedding.Length);
            Assert.NotEqual(42f, (float)network.GetEmbedding(3)[0]);
            Assert.Equal(VOCABULARY_SIZE, network.Embeddings.Length);
        }

        [Fact]
        public void Embed_ReturnsOneVectorPerToken()
        {
            IEmbeddingNetwork network = CreateNetwork(CreateSettings(), 0);

            Quad[][] vectors = network.Embed(new[] { 1, 1, 4 });

            Assert.Equal(3, vectors.Length);
            Assert.All(vectors, v => Assert.Equal(8, v.Length));
            Assert.Equal(vectors[0].Select(v => (float)v), vectors[1].Select(v => (float)v));
            Assert.NotSame(vectors[0], vectors[1]);
        }

        [Fact]
        public void CosineSimilarity_OfTokenWithItself_IsOne()
        {
            IEmbeddingNetwork network = CreateNetwork(CreateSettings(), 0);

            Assert.Equal(1f, (float)network.CosineSimilarity(2, 2), 5);
        }

        [Fact]
        public void Build_RegistersOneOptimizerPerWeight()
        {
            IEmbeddingNetwork network = CreateNetwork(CreateSettings(), 0);

            Assert.Equal(2 * VOCABULARY_SIZE * 8, network.Optimizers.Count);
        }

        [Fact]
        public void Propagate_TokenOutsideVocabulary_Throws()
        {
            IEmbeddingNetwork network = CreateNetwork(CreateSettings(), 0);

            Assert.Throws<ArgumentOutOfRangeException>(() => network.Propagate(new[] { VOCABULARY_SIZE }, new[] { 0 }));
            Assert.Throws<ArgumentOutOfRangeException>(() => network.Propagate(new[] { 0 }, new[] { -1 }));
            Assert.Throws<ArgumentOutOfRangeException>(() => network.GetEmbedding(VOCABULARY_SIZE));
        }

        [Fact]
        public void Backpropagate_WithoutPropagate_Throws()
        {
            IEmbeddingNetwork network = CreateNetwork(CreateSettings(), 0);

            Assert.Throws<InvalidOperationException>(() => network.Backpropagate(new Quad[] { 1 }));
        }

        [Fact]
        public void Backpropagate_WrongTargetsCount_Throws()
        {
            IEmbeddingNetwork network = CreateNetwork(CreateSettings(), 0);
            network.Propagate(new[] { 0 }, new[] { 1, 2 });

            Assert.Throws<ArgumentException>(() => network.Backpropagate(new Quad[] { 1 }));
        }

        [Fact]
        public void Build_InvalidSettings_Throw()
        {
            Word2VecSettings tooSmallVocabulary = CreateSettings();
            tooSmallVocabulary.VocabularySize = 1;
            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildWord2Vec(tooSmallVocabulary));

            Word2VecSettings noNegatives = CreateSettings();
            noNegatives.NegativeSamplesCount = 0;
            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildWord2Vec(noNegatives));

            Word2VecSettings zeroWindow = CreateSettings();
            zeroWindow.WindowSize = 0;
            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildWord2Vec(zeroWindow));

            Assert.Throws<ArgumentNullException>(() => NetworkFactory.BuildWord2Vec(null));
        }
    }
}
