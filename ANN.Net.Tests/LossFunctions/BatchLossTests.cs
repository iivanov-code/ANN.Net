using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Models;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.LossFunctions;
using ANN.Net.Matrix;
using Xunit;

namespace ANN.Net.Tests.LossFunctions
{
    /// <summary>
    /// Losses of whole batches: ranking, embedding pairs, triplets, Q-learning and policy gradients, and the training
    /// step on a loss of all outputs together
    /// </summary>
    public class BatchLossTests
    {
        private static Quad[] V(params double[] values) => values.Select(x => (Quad)x).ToArray();

        private static Quad[] Random(Random random, int size) => Enumerable.Range(0, size).Select(_ => (Quad)((random.NextDouble() * 2) - 1)).ToArray();

        private static void Rate(IVectorNetwork network, float rate)
        {
            foreach (LearningRateOptimizer optimizer in network.Optimizers) optimizer.LearningRate = rate;
        }

        private static IMatrixNetwork Matrix(ushort inputs, ushort hidden, ushort outputs, ActivationTypes outputActivation, OptimizerTypes optimizer = OptimizerTypes.Adam, int seed = 1,
                                             bool softMax = false, LossFunctionTypes loss = LossFunctionTypes.Squared) =>
            NetworkFactory.BuildMatrixFFN(new NetworkSettings
            {
                Seed = seed, LossType = loss, OutputApplySoftMax = softMax,
                InputSettings = new InputLayerSettings { NeuronsCount = inputs, ActivationType = ActivationTypes.Identity, HasBiasNeuron = true },
                OutputSettings = new OutputLayerSettings { NeuronsCount = outputs, ActivationType = outputActivation, OptimizerType = optimizer, InitializationType = WeightInitType.Xavier }
            }, new RecurrentLayerSettings
            {
                NeuronsCount = hidden, LayersCount = 1, ActivationType = ActivationTypes.HyperbolicTangens, PrevLayerConnectionType = LayerConnectionType.FullyConnected,
                HasBiasNeuron = true, OptimizerType = optimizer, InitializationType = WeightInitType.Xavier
            });

        /// <summary>
        /// Compares the analytic gradient with central differences of the loss, value by value
        /// </summary>
        private static void AssertGradient(Func<double> loss, IReadOnlyList<Quad[]> values, IReadOnlyList<Quad[]> gradients, string name)
        {
            const double step = 1e-3;
            for (int v = 0; v < values.Count; v++)
            {
                for (int k = 0; k < values[v].Length; k++)
                {
                    Quad original = values[v][k];
                    values[v][k] = (Quad)(original + step);
                    double plus = loss();
                    values[v][k] = (Quad)(original - step);
                    double minus = loss();
                    values[v][k] = original;

                    double numeric = (plus - minus) / (2 * step);
                    Assert.True(Math.Abs(numeric - gradients[v][k]) <= 2e-3 * Math.Max(1, Math.Abs(numeric)),
                                $"{name} vector {v} value {k}: numeric {numeric} vs analytic {(float)gradients[v][k]}");
                }
            }
        }

        // ---- values: the formulas of the image, written out independently, and small cases worked out by hand

        private static double RankingFormula(RankingLossTypes type, double[] scores, double[] relevance, double margin)
        {
            int n = scores.Length;
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double indicator = relevance[i] > relevance[j] ? 1 : 0;
                    double d = (1 - (2 * indicator)) * (scores[i] - scores[j]);
                    sum += type switch
                    {
                        RankingLossTypes.MarginRanking => Math.Max(0, d + margin),
                        RankingLossTypes.SoftPairwiseHinge => Math.Log(1 + Math.Exp(d + margin)),
                        _ => Math.Log(1 + Math.Exp(d)),
                    };
                }
            }

            return sum / (n * n);
        }

        private static readonly double[] Scores = { 0.3, -0.2, 0.9, 0.1, 0.35 };
        private static readonly double[] Relevance = { 2, 0, 1, 1, 2 };

        [Theory]
        [InlineData(RankingLossTypes.MarginRanking)]
        [InlineData(RankingLossTypes.SoftPairwiseHinge)]
        [InlineData(RankingLossTypes.PairwiseLogistic)]
        public void RankingLosses_FollowTheirFormulas_AndTheirGradients(RankingLossTypes type)
        {
            var loss = new PairwiseRankingLoss(type, 0.7f);
            Quad[] scores = V(Scores);
            Quad[] relevance = V(Relevance);
            Assert.Equal(RankingFormula(type, Scores, Relevance, 0.7), loss.CalculateLoss(scores, relevance), 5);

            Quad[] gradient = loss.CalculateGradient(scores, relevance);
            AssertGradient(() => loss.CalculateLoss(scores, relevance), new[] { scores }, new[] { gradient }, type.ToString());
        }

        [Fact]
        public void MarginRanking_WorkedOutByHand()
        {
            // pairs (0,0) and (1,1) give the margin 1 each; the ordered pairs are exactly one margin apart and give 0
            Assert.Equal(0.5, new PairwiseRankingLoss(RankingLossTypes.MarginRanking).CalculateLoss(V(1, 0), V(1, 0)), 6);

            // swapped scores: (0,1) gives -(0 - 1) + 1 = 2, (1,0) gives (1 - 0) + 1 = 2
            Assert.Equal(1.5, new PairwiseRankingLoss(RankingLossTypes.MarginRanking).CalculateLoss(V(0, 1), V(1, 0)), 6);
            Assert.Equal(0, new PairwiseRankingLoss(RankingLossTypes.MarginRanking).CalculateLoss(Array.Empty<Quad>(), Array.Empty<Quad>()), 6);
        }

        [Fact]
        public void EmbeddingLosses_WorkedOutByHand()
        {
            var cosine = new EmbeddingPairLoss(EmbeddingLossTypes.CosineEmbedding, 0.5f);
            // orthogonal and alike: 1 - 0; parallel and different: max(0, 1 - 0.5); orthogonal and different: 0
            Assert.Equal(1, cosine.CalculateLoss(new[] { V(1, 0) }, new[] { V(0, 1) }, new[] { 1 }), 6);
            Assert.Equal(0.5, cosine.CalculateLoss(new[] { V(1, 1) }, new[] { V(2, 2) }, new[] { -1 }), 6);
            Assert.Equal(0.5, cosine.CalculateLoss(new[] { V(1, 0), V(1, 1), V(1, 0) }, new[] { V(0, 1), V(2, 2), V(0, 3) }, new[] { 1, -1, -1 }), 6);

            var hinge = new EmbeddingPairLoss(EmbeddingLossTypes.HingeEmbedding, 3);
            // squared distance 1: alike gives 1, different gives max(0, 3 - 1)
            Assert.Equal(1, hinge.CalculateLoss(new[] { V(1, 0) }, new[] { V(0, 0) }, new[] { 1 }), 6);
            Assert.Equal(2, hinge.CalculateLoss(new[] { V(1, 0) }, new[] { V(0, 0) }, new[] { -1 }), 6);
            Assert.Equal(0, hinge.CalculateLoss(new[] { V(2, 0) }, new[] { V(0, 0) }, new[] { -1 }), 6);
        }

        [Fact]
        public void TripletAndReinforcementLosses_WorkedOutByHand()
        {
            var triplet = new TripletMarginLoss(1);
            // d(a, p) = 1: a negative at squared distance 4 is far enough, one at 1 is not
            Assert.Equal(0, triplet.CalculateLoss(new[] { V(0, 0) }, new[] { V(1, 0) }, new[] { V(0, 2) }), 6);
            Assert.Equal(1, triplet.CalculateLoss(new[] { V(0, 0) }, new[] { V(1, 0) }, new[] { V(0, 1) }), 6);

            // (2 - 3)² and (0 - 0.5)², averaged
            Assert.Equal(0.625, new QValueLoss(0.9f).CalculateLoss(new[] { V(1, 2), V(0, 4) }, new[] { 1, 0 }, V(3, 0.5)), 6);

            // -(2 ln 0.25 + -1 ln 0.5) / 2
            Assert.Equal(-((2 * Math.Log(0.25)) - Math.Log(0.5)) / 2, new PolicyGradientLoss().CalculateLoss(new[] { V(0.25, 0.75), V(0.5, 0.5) }, new[] { 0, 1 }, V(2, -1)), 5);
        }

        public static TheoryData<EmbeddingLossTypes, float> EmbeddingCases => new TheoryData<EmbeddingLossTypes, float>
        {
            { EmbeddingLossTypes.CosineEmbedding, 0.1f },
            { EmbeddingLossTypes.CosineEmbedding, -0.4f },
            { EmbeddingLossTypes.HingeEmbedding, 3 },
            { EmbeddingLossTypes.HingeEmbedding, 0.5f },
        };

        [Theory]
        [MemberData(nameof(EmbeddingCases))]
        public void EmbeddingPairGradients_MatchFiniteDifferences(EmbeddingLossTypes type, float margin)
        {
            var random = new Random(4);
            List<Quad[]> first = Enumerable.Range(0, 6).Select(_ => Random(random, 3)).ToList();
            List<Quad[]> second = Enumerable.Range(0, 6).Select(_ => Random(random, 3)).ToList();
            int[] labels = { 1, -1, 1, -1, -1, 1 };
            var loss = new EmbeddingPairLoss(type, margin);

            (Quad[][] firstGradient, Quad[][] secondGradient) = loss.CalculateGradient(first, second, labels);
            Assert.Contains(firstGradient.Concat(secondGradient).SelectMany(x => x), g => g != 0);
            AssertGradient(() => loss.CalculateLoss(first, second, labels), first.Concat(second).ToList(), firstGradient.Concat(secondGradient).ToList(), type.ToString());
        }

        [Fact]
        public void TripletAndReinforcementGradients_MatchFiniteDifferences()
        {
            var random = new Random(5);
            List<Quad[]> anchors = Enumerable.Range(0, 5).Select(_ => Random(random, 3)).ToList();
            List<Quad[]> positives = Enumerable.Range(0, 5).Select(_ => Random(random, 3)).ToList();
            List<Quad[]> negatives = Enumerable.Range(0, 5).Select(_ => Random(random, 3)).ToList();
            var triplet = new TripletMarginLoss(0.8f);
            var (a, p, n) = triplet.CalculateGradient(anchors, positives, negatives);
            Assert.Contains(a.SelectMany(x => x), g => g != 0);
            AssertGradient(() => triplet.CalculateLoss(anchors, positives, negatives), anchors.Concat(positives).Concat(negatives).ToList(),
                           a.Concat(p).Concat(n).ToList(), "triplet");

            List<Quad[]> values = Enumerable.Range(0, 4).Select(_ => Random(random, 3)).ToList();
            int[] actions = { 0, 2, 1, 2 };
            Quad[] targets = V(0.5, -0.3, 1, 0.1);
            var q = new QValueLoss(0.9f);
            AssertGradient(() => q.CalculateLoss(values, actions, targets), values, q.CalculateGradient(values, actions, targets), "Q value");

            List<Quad[]> probabilities = new[] { V(0.2, 0.3, 0.5), V(0.6, 0.1, 0.3), V(0.25, 0.25, 0.5) }.ToList();
            int[] taken = { 2, 0, 1 };
            Quad[] returns = V(1.5, -0.5, 2);
            var policy = new PolicyGradientLoss();
            AssertGradient(() => policy.CalculateLoss(probabilities, taken, returns), probabilities, policy.CalculateGradient(probabilities, taken, returns), "policy");
        }

        [Fact]
        public void InvalidArgumentsAreRejected()
        {
            Assert.Throws<ArgumentException>(() => new PairwiseRankingLoss((RankingLossTypes)9));
            Assert.Throws<ArgumentException>(() => new PairwiseRankingLoss(RankingLossTypes.MarginRanking, float.NaN));
            Assert.Throws<ArgumentException>(() => new PairwiseRankingLoss(RankingLossTypes.MarginRanking).CalculateLoss(V(1, 2), V(1)));
            Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingPairLoss(EmbeddingLossTypes.CosineEmbedding, 1.5f));
            Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingPairLoss(EmbeddingLossTypes.HingeEmbedding, -1));
            Assert.Throws<ArgumentException>(() => new EmbeddingPairLoss(EmbeddingLossTypes.HingeEmbedding, 1).CalculateLoss(new[] { V(1) }, new[] { V(1) }, new[] { 0 }));
            Assert.Throws<ArgumentException>(() => new EmbeddingPairLoss(EmbeddingLossTypes.HingeEmbedding, 1).CalculateLoss(new[] { V(1) }, new[] { V(1, 2) }, new[] { 1 }));
            Assert.Throws<ArgumentException>(() => new TripletMarginLoss().CalculateLoss(new[] { V(1) }, new[] { V(1) }, new[] { V(1, 2) }));
            Assert.Throws<ArgumentOutOfRangeException>(() => new QValueLoss(1.5f));
            Assert.Throws<ArgumentOutOfRangeException>(() => new QValueLoss(0.9f).CalculateLoss(new[] { V(1, 2) }, new[] { 2 }, V(0)));
            Assert.Throws<ArgumentOutOfRangeException>(() => new PolicyGradientLoss().CalculateLoss(new[] { V(1) }, new[] { -1 }, V(0)));

            IMatrixNetwork twoOutputs = Matrix(2, 3, 2, ActivationTypes.Identity);
            Assert.Throws<ArgumentException>(() => new PairwiseRankingLoss(RankingLossTypes.PairwiseLogistic).Train(twoOutputs, new[] { V(1, 2) }, V(1)));
        }

        [Fact]
        public void QTargets_AreRewardPlusDiscountedBestNextValue()
        {
            IMatrixNetwork network = Matrix(2, 4, 3, ActivationTypes.Identity);
            Quad[] next = V(0.4, -0.6);
            double best = network.Propagate(next).Max();
            Quad[] targets = new QValueLoss(0.5f).Targets(network, new[]
            {
                new Transition { State = V(0, 0), Action = 1, Reward = 2, NextState = next },
                new Transition { State = V(0, 0), Action = 1, Reward = 2, NextState = null },
            });

            Assert.Equal(2 + (0.5 * best), targets[0], 5);
            Assert.Equal(2, targets[1], 6);
        }

        // ---- the training step on a loss of all outputs together

        /// <summary>
        /// With plain gradient descent at rate 1 every weight changes by the gradient of the batch loss, which is the sum
        /// over the samples: compared here with finite differences of the triplet loss of three embedded inputs
        /// </summary>
        [Fact]
        public void TrainOutputs_MovesEveryWeightByTheGradientOfTheBatchLoss()
        {
            var network = (MatrixNetwork)Matrix(3, 4, 2, ActivationTypes.HyperbolicTangens, OptimizerTypes.Normal, seed: 7);
            var loss = new TripletMarginLoss(2);
            var random = new Random(8);
            List<Quad[]> anchors = Enumerable.Range(0, 3).Select(_ => Random(random, 3)).ToList();
            List<Quad[]> positives = Enumerable.Range(0, 3).Select(_ => Random(random, 3)).ToList();
            List<Quad[]> negatives = Enumerable.Range(0, 3).Select(_ => Random(random, 3)).ToList();

            List<Quad[]> Embed(List<Quad[]> inputs) => inputs.Select(x => (Quad[])network.Propagate(x).Clone()).ToList();
            double LossOf() => loss.CalculateLoss(Embed(anchors), Embed(positives), Embed(negatives));

            var tensors = network.Layers.SelectMany(x => new[] { x.Weights, x.Bias }).ToList();
            var numeric = new List<double[]>();
            foreach (Quad[] tensor in tensors)
            {
                var gradient = new double[tensor.Length];
                for (int i = 0; i < tensor.Length; i++)
                {
                    Quad original = tensor[i];
                    tensor[i] = (Quad)(original + 1e-3);
                    double plus = LossOf();
                    tensor[i] = (Quad)(original - 1e-3);
                    double minus = LossOf();
                    tensor[i] = original;
                    gradient[i] = (plus - minus) / 2e-3;
                }

                numeric.Add(gradient);
            }

            double before = LossOf();
            Assert.True(before > 0);
            List<Quad[]> weights = tensors.Select(x => (Quad[])x.Clone()).ToList();
            Rate(network, 1);
            Assert.Equal(before, loss.Train(network, anchors, positives, negatives), 5);

            for (int k = 0; k < tensors.Count; k++)
            {
                for (int i = 0; i < tensors[k].Length; i++)
                {
                    double analytic = weights[k][i] - tensors[k][i];
                    Assert.True(Math.Abs(numeric[k][i] - analytic) <= 1e-2 * Math.Max(1, Math.Abs(numeric[k][i])),
                                $"tensor {k} value {i}: numeric {numeric[k][i]} vs analytic {analytic}");
                }
            }
        }

        private static int[] Bits(IEnumerable<Quad> values) => values.Select(x => BitConverter.SingleToInt32Bits(x)).ToArray();

        private static IConvolutionalNetwork Convolutional() => NetworkFactory.BuildConvolutional(new ConvolutionalNetworkSettings
        {
            InputChannels = 2, InputHeight = 6, InputWidth = 6, OutputDimension = 3, OutputApplySoftMax = true, Seed = 2, OptimizerType = OptimizerTypes.Adam,
            Layers =
            {
                new ConvolutionLayerSettings { Filters = 4, KernelSize = 3, Padding = 1, UseBias = false, ActivationType = ActivationTypes.Identity },
                new BatchNormalizationLayerSettings { ActivationType = ActivationTypes.RELU },
                new GlobalAveragePoolingLayerSettings(),
                new DenseLayerSettings { NeuronsCount = 5, ActivationType = ActivationTypes.HyperbolicTangens }
            }
        });

        /// <summary>
        /// The cross entropy of every sample, as a loss of the whole batch: its gradient through the soft-max is
        /// (predicted - target) / N, the mean that TrainBatch trains on
        /// </summary>
        private static BatchLoss CrossEntropy(IReadOnlyList<Quad[]> targets) => (outputs, gradients) =>
        {
            double loss = 0;
            for (int i = 0; i < outputs.Count; i++)
            {
                gradients[i] = outputs[i].Select((p, k) => (Quad)(-targets[i][k] / ((double)p * outputs.Count))).ToArray();
                loss -= outputs[i].Select((p, k) => targets[i][k] * Math.Log(p)).Sum() / outputs.Count;
            }

            return (Quad)loss;
        };

        /// <summary>
        /// Batch normalization propagates every image twice, for the outputs and for the gradients, but advances its
        /// running averages once per image, as TrainBatch does
        /// </summary>
        [Fact]
        public void TrainOutputs_OnConvolutionalNetworks_TrainsLikeTrainBatch()
        {
            IConvolutionalNetwork batched = Convolutional(), outputs = Convolutional();
            Rate(batched, 0.01f);
            Rate(outputs, 0.01f);

            var random = new Random(3);
            for (int step = 0; step < 5; step++)
            {
                List<Quad[]> images = Enumerable.Range(0, 6).Select(_ => Random(random, 72)).ToList();
                List<Quad[]> targets = Enumerable.Range(0, 6).Select(i => Enumerable.Range(0, 3).Select(k => (Quad)(k == (i + step) % 3 ? 1 : 0)).ToArray()).ToList();
                float expected = batched.TrainBatch(images, targets, new BatchOptions { ChunkSize = 2 });
                Assert.Equal(expected, (float)outputs.TrainOutputs(images, CrossEntropy(targets), new BatchOptions { ChunkSize = 2 }), 4);
            }

            batched.Training = outputs.Training = false;
            Quad[] probe = Random(random, 72);
            Assert.Equal(batched.Propagate(probe).Select(x => (float)x), outputs.Propagate(probe).Select(x => (float)x), new FloatTolerance(1e-4f));
        }

        private sealed class FloatTolerance : IEqualityComparer<float>
        {
            private readonly float tolerance;

            public FloatTolerance(float tolerance) => this.tolerance = tolerance;

            public bool Equals(float x, float y) => Math.Abs(x - y) <= tolerance;

            public int GetHashCode(float obj) => 0;
        }

        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(4)]
        public void TrainOutputs_DoesNotDependOnTheNumberOfThreads(int threads)
        {
            IMatrixNetwork reference = Matrix(3, 6, 2, ActivationTypes.Identity, seed: 4), other = Matrix(3, 6, 2, ActivationTypes.Identity, seed: 4);
            Rate(reference, 0.01f);
            Rate(other, 0.01f);
            var loss = new EmbeddingPairLoss(EmbeddingLossTypes.HingeEmbedding, 2);
            var random = new Random(6);
            for (int step = 0; step < 4; step++)
            {
                List<Quad[]> first = Enumerable.Range(0, 9).Select(_ => Random(random, 3)).ToList();
                List<Quad[]> second = Enumerable.Range(0, 9).Select(_ => Random(random, 3)).ToList();
                int[] labels = Enumerable.Range(0, 9).Select(i => i % 2 == 0 ? 1 : -1).ToArray();
                Assert.Equal((float)loss.Train(reference, first, second, labels, new BatchOptions { ChunkSize = 3, MaxDegreeOfParallelism = 1 }),
                             (float)loss.Train(other, first, second, labels, new BatchOptions { ChunkSize = 3, MaxDegreeOfParallelism = threads }));
            }

            Quad[] probe = V(0.1, 0.2, -0.3);
            Assert.Equal(Bits(reference.Propagate(probe)), Bits(other.Propagate(probe)));
        }

        /// <summary>
        /// Networks without batch training apply the samples one after the other; with one sample that is exactly the
        /// batch step of a matrix network
        /// </summary>
        [Fact]
        public void TrainOutputs_OfOneSample_IsTheSameOnNeuronAndMatrixNetworks()
        {
            NetworkSettings Settings() => new NetworkSettings
            {
                Seed = 5, IdentityGenerator = new IdentityGenerators.IncrementalGenerator(),
                InputSettings = new InputLayerSettings { NeuronsCount = 3, ActivationType = ActivationTypes.Identity, HasBiasNeuron = true },
                OutputSettings = new OutputLayerSettings { NeuronsCount = 2, ActivationType = ActivationTypes.Sigmoid, OptimizerType = OptimizerTypes.Adam }
            };

            var neurons = (IVectorNetwork)NetworkFactory.BuildFFN(Settings(), (RecurrentLayerSettings)null);
            IMatrixNetwork matrix = NetworkFactory.BuildMatrixFFN(Settings());
            Rate(neurons, 0.05f);
            Rate(matrix, 0.05f);

            var policy = new PolicyGradientLoss();
            for (int step = 0; step < 10; step++)
            {
                Quad[] state = V(step % 3, 0.5, -0.25);
                Assert.Equal((float)policy.Train(neurons, new[] { state }, new[] { step % 2 }, V(1)), (float)policy.Train(matrix, new[] { state }, new[] { step % 2 }, V(1)));
            }

            Assert.Equal(Bits(neurons.Propagate(V(1, 1, 1))), Bits(matrix.Propagate(V(1, 1, 1))));

            // several samples one after the other still lower the loss on them
            List<Quad[]> states = Enumerable.Range(0, 4).Select(i => V(i, 1, -i)).ToList();
            int[] actions = { 0, 1, 0, 1 };
            Quad[] returns = V(1, 1, 1, 1);
            float first = policy.Train(neurons, states, actions, returns);
            for (int step = 0; step < 50; step++) policy.Train(neurons, states, actions, returns);
            Assert.True(policy.Train(neurons, states, actions, returns) < first);
        }

        [Fact]
        public void TrainOutputs_RejectsMissingGradients()
        {
            IMatrixNetwork matrix = Matrix(2, 3, 1, ActivationTypes.Identity);
            var neurons = (IVectorNetwork)NetworkFactory.BuildFFN(new NetworkSettings
            {
                Seed = 1,
                InputSettings = new InputLayerSettings { NeuronsCount = 2, ActivationType = ActivationTypes.Identity },
                OutputSettings = new OutputLayerSettings { NeuronsCount = 1, ActivationType = ActivationTypes.Identity }
            }, (RecurrentLayerSettings)null);

            foreach (IVectorNetwork network in new[] { matrix, neurons })
            {
                Assert.Throws<InvalidOperationException>(() => network.TrainOutputs(new[] { V(1, 2) }, (_, _) => 0));
            }
        }

        // ---- every loss teaches the network what it should

        [Theory]
        [InlineData(RankingLossTypes.MarginRanking)]
        [InlineData(RankingLossTypes.SoftPairwiseHinge)]
        [InlineData(RankingLossTypes.PairwiseLogistic)]
        public void RankingLosses_LearnToOrderByRelevance(RankingLossTypes type)
        {
            static Quad Relevance(Quad[] x) => (Quad)Math.Floor((x[0] + (2 * x[1]) + 3) * 5 / 6);

            IMatrixNetwork network = Matrix(2, 8, 1, ActivationTypes.Identity);
            Rate(network, 0.01f);
            var loss = new PairwiseRankingLoss(type, 0.5f);
            var random = new Random(1);
            for (int step = 0; step < 300; step++)
            {
                List<Quad[]> inputs = Enumerable.Range(0, 32).Select(_ => Random(random, 2)).ToList();
                loss.Train(network, inputs, inputs.Select(Relevance).ToList());
            }

            List<Quad[]> test = Enumerable.Range(0, 60).Select(_ => Random(random, 2)).ToList();
            double[] scores = test.Select(x => (double)network.Propagate(x)[0]).ToArray();
            int right = 0, pairs = 0;
            for (int i = 0; i < test.Count; i++)
            {
                for (int j = 0; j < test.Count; j++)
                {
                    if (Relevance(test[i]) <= Relevance(test[j])) continue;
                    pairs++;
                    if (scores[i] > scores[j]) right++;
                }
            }

            Assert.True(right > 0.95 * pairs, $"{right} of {pairs} pairs in order");
        }

        /// <summary>
        /// Three clusters of 4-dimensional points, embedded in 2 dimensions: points of a cluster should end up closer to
        /// each other than to points of other clusters
        /// </summary>
        private static (Quad[] Point, int Cluster) Clustered(Random random)
        {
            Quad[][] centers = { V(1, 0, 0, 1), V(-1, 1, 0, 0), V(0, -1, 1, -1) };
            int cluster = random.Next(3);
            return (centers[cluster].Select(x => (Quad)(x + (0.3 * ((random.NextDouble() * 2) - 1)))).ToArray(), cluster);
        }

        private static double Separation(IVectorNetwork network, Func<Quad[], Quad[], double> distance)
        {
            var random = new Random(99);
            int right = 0;
            for (int t = 0; t < 200; t++)
            {
                (Quad[] a, int cluster) = Clustered(random);
                (Quad[] p, int positive) = Clustered(random);
                (Quad[] n, int negative) = Clustered(random);
                while (positive != cluster) (p, positive) = Clustered(random);
                while (negative == cluster) (n, negative) = Clustered(random);

                Quad[] ea = (Quad[])network.Propagate(a).Clone(), ep = (Quad[])network.Propagate(p).Clone(), en = (Quad[])network.Propagate(n).Clone();
                if (distance(ea, ep) < distance(ea, en)) right++;
            }

            return right / 200.0;
        }

        private static double Squared(Quad[] a, Quad[] b) => a.Select((x, k) => Math.Pow((double)x - b[k], 2)).Sum();

        private static double NegativeCosine(Quad[] a, Quad[] b) =>
            -a.Select((x, k) => (double)x * b[k]).Sum() / Math.Sqrt(a.Sum(x => (double)x * x) * b.Sum(x => (double)x * x));

        [Fact]
        public void TripletLoss_SeparatesClusters()
        {
            IMatrixNetwork network = Matrix(4, 8, 2, ActivationTypes.Identity, seed: 2);
            Rate(network, 0.01f);
            var loss = new TripletMarginLoss(1);
            var random = new Random(2);
            for (int step = 0; step < 200; step++)
            {
                var anchors = new List<Quad[]>();
                var positives = new List<Quad[]>();
                var negatives = new List<Quad[]>();
                for (int t = 0; t < 16; t++)
                {
                    (Quad[] a, int cluster) = Clustered(random);
                    (Quad[] p, int positive) = Clustered(random);
                    (Quad[] n, int negative) = Clustered(random);
                    while (positive != cluster) (p, positive) = Clustered(random);
                    while (negative == cluster) (n, negative) = Clustered(random);
                    anchors.Add(a);
                    positives.Add(p);
                    negatives.Add(n);
                }

                loss.Train(network, anchors, positives, negatives);
            }

            Assert.True(Separation(network, Squared) > 0.95);
        }

        [Theory]
        [InlineData(EmbeddingLossTypes.HingeEmbedding, 4)]
        [InlineData(EmbeddingLossTypes.CosineEmbedding, 0)]
        public void EmbeddingPairLosses_SeparateClusters(EmbeddingLossTypes type, float margin)
        {
            IMatrixNetwork network = Matrix(4, 8, 2, ActivationTypes.Identity, seed: 3);
            Rate(network, 0.01f);
            var loss = new EmbeddingPairLoss(type, margin);
            var random = new Random(3);
            for (int step = 0; step < 200; step++)
            {
                var first = new List<Quad[]>();
                var second = new List<Quad[]>();
                var labels = new List<int>();
                for (int t = 0; t < 16; t++)
                {
                    (Quad[] a, int ca) = Clustered(random);
                    (Quad[] b, int cb) = Clustered(random);
                    first.Add(a);
                    second.Add(b);
                    labels.Add(ca == cb ? 1 : -1);
                }

                loss.Train(network, first, second, labels);
            }

            Assert.True(Separation(network, type == EmbeddingLossTypes.HingeEmbedding ? Squared : NegativeCosine) > 0.95);
        }

        /// <summary>
        /// A corridor of five cells: moving right from cell 3 reaches the goal in cell 4 with reward 1 and ends the
        /// episode, nothing else gives a reward. The best values are Q(s, right) = γ^(3 - s) and
        /// Q(s, left) = γ^(5 - s), γ⁴ in cell 0 where moving left stays put
        /// </summary>
        [Fact]
        public void QLearning_FindsTheValuesOfACorridor()
        {
            static Quad[] Cell(int s) => Enumerable.Range(0, 5).Select(i => (Quad)(i == s ? 1 : 0)).ToArray();

            var transitions = new List<Transition>();
            for (int s = 0; s < 4; s++)
            {
                transitions.Add(new Transition { State = Cell(s), Action = 0, Reward = 0, NextState = Cell(Math.Max(0, s - 1)) });
                transitions.Add(new Transition { State = Cell(s), Action = 1, Reward = s == 3 ? 1 : 0, NextState = s == 3 ? null : Cell(s + 1) });
            }

            IMatrixNetwork network = Matrix(5, 12, 2, ActivationTypes.Identity, seed: 4);
            Rate(network, 0.005f);
            var loss = new QValueLoss(0.9f);
            for (int step = 0; step < 3000; step++) loss.Train(network, transitions);

            for (int s = 0; s < 4; s++)
            {
                Quad[] q = network.Propagate(Cell(s));
                Assert.True(q[1] > q[0], $"cell {s}: right {(float)q[1]}, left {(float)q[0]}");
                Assert.Equal(Math.Pow(0.9, 3 - s), q[1], 0.05);
                Assert.Equal(Math.Pow(0.9, s == 0 ? 4 : 5 - s), q[0], 0.08);
            }
        }

        /// <summary>
        /// Two situations with three possible actions each; only one action pays off in each. Actions are drawn from the
        /// policy, and the reward minus the mean reward of the batch tells the policy which to prefer
        /// </summary>
        [Fact]
        public void PolicyGradient_LearnsTheBestActionOfEverySituation()
        {
            IConvolutionalNetwork network = NetworkFactory.BuildConvolutional(new ConvolutionalNetworkSettings
            {
                InputChannels = 2, InputHeight = 1, InputWidth = 1, OutputDimension = 3, OutputApplySoftMax = true, Seed = 1, OptimizerType = OptimizerTypes.Adam,
                Layers = { new DenseLayerSettings { NeuronsCount = 6, ActivationType = ActivationTypes.HyperbolicTangens } }
            });
            Rate(network, 0.02f);

            int[] best = { 2, 0 };
            var loss = new PolicyGradientLoss();
            var random = new Random(5);
            for (int step = 0; step < 300; step++)
            {
                var states = new List<Quad[]>();
                var actions = new List<int>();
                var rewards = new List<double>();
                for (int t = 0; t < 16; t++)
                {
                    int situation = random.Next(2);
                    Quad[] state = V(situation == 0 ? 1 : 0, situation);
                    Quad[] policy = network.Propagate(state);
                    double draw = random.NextDouble();
                    int action = 0;
                    while (action < 2 && draw > policy[action])
                    {
                        draw -= policy[action];
                        action++;
                    }

                    states.Add(state);
                    actions.Add(action);
                    rewards.Add(action == best[situation] ? 1 : 0);
                }

                double baseline = rewards.Average();
                loss.Train(network, states, actions, rewards.Select(r => (Quad)(r - baseline)).ToList());
            }

            Assert.True(network.Propagate(V(1, 0))[2] > 0.9);
            Assert.True(network.Propagate(V(0, 1))[0] > 0.9);
        }
    }
}
