using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using Xunit;

namespace ANN.Net.Tests.LanguageModels
{
    public class LanguageModelTests
    {
        private const int VOCABULARY_SIZE = 6;
        private const int MAX_SEQUENCE_LENGTH = 8;

        private static readonly WordTokenizer TOKENIZER = new WordTokenizer();

        private static TransformerSettings TransformerSettings(int inputDimension = VOCABULARY_SIZE, int outputDimension = VOCABULARY_SIZE, bool applySoftMax = true)
        {
            return new TransformerSettings
            {
                InputDimension = (ushort)inputDimension,
                ModelDimension = 8,
                HeadsCount = 2,
                FeedForwardDimension = 16,
                LayersCount = 2,
                OutputDimension = (ushort)outputDimension,
                MaxSequenceLength = MAX_SEQUENCE_LENGTH,
                UseCausalMask = true,
                OutputApplySoftMax = applySoftMax,
                OptimizerType = OptimizerTypes.Normal,
                InitializationType = WeightInitType.Xavier
            };
        }

        private static ISequenceNetwork Transformer(float learningRate, TransformerSettings settings = null)
        {
            ISequenceNetwork network = NetworkFactory.BuildTransformer(settings ?? TransformerSettings());
            foreach (LearningRateOptimizer optimizer in network.Optimizers)
            {
                optimizer.LearningRate = learningRate;
            }

            return network;
        }

        private static LanguageModelSettings ModelSettings(int contextLength = 6)
        {
            return new LanguageModelSettings
            {
                ContextLength = (ushort)contextLength,
                ShuffleWindows = false,
                Seed = 1
            };
        }

        private static ILanguageModel ScriptedModel(int nextToken, float probability = 1f, int contextLength = 4)
        {
            ScriptedSequenceNetwork network = new ScriptedSequenceNetwork(VOCABULARY_SIZE, MAX_SEQUENCE_LENGTH, nextToken, probability);
            return NetworkFactory.BuildLanguageModel(ModelSettings(contextLength), TOKENIZER, network);
        }

        /// <summary>
        /// The periodic language "a b c d a b c d ..."
        /// </summary>
        private static List<int[]> PeriodicCorpus()
        {
            return new List<int[]>
            {
                TOKENIZER.Encode("a b c d a b c d"),
                TOKENIZER.Encode("b c d a b c d a"),
                TOKENIZER.Encode("c d a b c d a b"),
                TOKENIZER.Encode("d a b c d a b c")
            };
        }

        [Fact]
        public void Build_OutputDimensionDiffersFromVocabulary_Throws()
        {
            ISequenceNetwork network = Transformer(0, TransformerSettings(outputDimension: 5));

            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildLanguageModel(ModelSettings(), TOKENIZER, network));
        }

        [Fact]
        public void Build_WithoutEmbeddings_InputDimensionMustBeVocabulary()
        {
            ISequenceNetwork network = Transformer(0, TransformerSettings(inputDimension: 4));

            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildLanguageModel(ModelSettings(), TOKENIZER, network));
        }

        [Fact]
        public void Build_ContextLongerThanNetwork_Throws()
        {
            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildLanguageModel(ModelSettings(MAX_SEQUENCE_LENGTH + 1), TOKENIZER, Transformer(0)));
        }

        [Fact]
        public void Build_WithoutSoftMax_Throws()
        {
            ISequenceNetwork network = Transformer(0, TransformerSettings(applySoftMax: false));

            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildLanguageModel(ModelSettings(), TOKENIZER, network));
        }

        [Fact]
        public void Build_UnknownSpecialToken_Throws()
        {
            LanguageModelSettings settings = ModelSettings();
            settings.BeginToken = "<start>";

            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildLanguageModel(settings, TOKENIZER, Transformer(0)));
        }

        [Fact]
        public void Build_NullArguments_Throw()
        {
            Assert.Throws<ArgumentNullException>(() => NetworkFactory.BuildLanguageModel(null, TOKENIZER, Transformer(0)));
            Assert.Throws<ArgumentNullException>(() => NetworkFactory.BuildLanguageModel(ModelSettings(), null, Transformer(0)));
            Assert.Throws<ArgumentNullException>(() => NetworkFactory.BuildLanguageModel(ModelSettings(), TOKENIZER, null));
        }

        [Fact]
        public void Build_WithEmbeddings_ChecksTheEmbeddingDimension()
        {
            IEmbeddingNetwork embeddings = NetworkFactory.BuildWord2Vec(new Word2VecSettings { VocabularySize = VOCABULARY_SIZE, EmbeddingDimension = 4, Seed = 1 });

            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildLanguageModel(ModelSettings(), TOKENIZER, Transformer(0), embeddings));

            ILanguageModel model = NetworkFactory.BuildLanguageModel(ModelSettings(), TOKENIZER, Transformer(0, TransformerSettings(inputDimension: 4)), embeddings);

            Assert.Same(embeddings, model.Embeddings);
            Quad loss = model.Train(PeriodicCorpus());
            Assert.True(loss > 0);
        }

        [Fact]
        public void Train_RepeatedEpochs_LearnsThePeriodicLanguage()
        {
            ILanguageModel model = NetworkFactory.BuildLanguageModel(ModelSettings(), TOKENIZER, Transformer(0.05f));
            List<int[]> corpus = PeriodicCorpus();

            Quad initialLoss = model.Evaluate(corpus);
            for (int epoch = 0; epoch < 150; epoch++)
            {
                model.Train(corpus);
            }

            Quad finalLoss = model.Evaluate(corpus);

            Assert.True(finalLoss < initialLoss / 2, $"Loss went from {initialLoss} to {finalLoss}");
            Assert.True(model.Perplexity(corpus) < 3, $"Perplexity {model.Perplexity(corpus)}");

            int[] continuation = model.Generate(TOKENIZER.Encode("a b c"), new GenerationSettings { Temperature = 0, MaxNewTokens = 1 });
            Assert.Equal(new[] { WordTokenizer.D }, continuation);
        }

        [Fact]
        public void Perplexity_IsTheExponentialOfTheLoss()
        {
            ILanguageModel model = ScriptedModel(WordTokenizer.A, probability: 0.5f);
            List<int[]> corpus = new List<int[]> { TOKENIZER.Encode("a a a") };

            Assert.Equal((float)Math.Exp(model.Evaluate(corpus)), (float)model.Perplexity(corpus), 3);
        }

        [Fact]
        public void Evaluate_ComputesCrossEntropyOfTheTargets()
        {
            ILanguageModel model = ScriptedModel(WordTokenizer.A, probability: 0.5f);

            // window: <bos> a a <eos>; targets a, a, <eos> with probabilities 0.5, 0.5, 0.1
            Quad loss = model.Evaluate(new[] { TOKENIZER.Encode("a a") });

            double expected = -(Math.Log(0.5) + Math.Log(0.5) + Math.Log(0.1)) / 3;
            Assert.Equal((float)expected, (float)loss, 4);
        }

        [Fact]
        public void Evaluate_DoesNotChangeTheWeights()
        {
            ILanguageModel model = NetworkFactory.BuildLanguageModel(ModelSettings(), TOKENIZER, Transformer(0.5f));
            int[] prompt = TOKENIZER.Encode("a b");

            Quad[] before = model.PredictNext(prompt);
            model.Evaluate(PeriodicCorpus());
            Quad[] after = model.PredictNext(prompt);

            Assert.Equal(before.Select(v => (float)v), after.Select(v => (float)v));
        }

        [Fact]
        public void Train_EmptyCorpus_ReturnsZero()
        {
            ILanguageModel model = NetworkFactory.BuildLanguageModel(ModelSettings(), TOKENIZER, Transformer(0.05f));

            Assert.Equal(0f, (float)model.Train(new List<int[]>()));
            Assert.Equal(0f, (float)model.Evaluate(new List<int[]>()));
        }

        [Fact]
        public void PredictNext_ReturnsDistributionOverTheVocabulary()
        {
            ILanguageModel model = NetworkFactory.BuildLanguageModel(ModelSettings(), TOKENIZER, Transformer(0));

            Quad[] distribution = model.PredictNext(TOKENIZER.Encode("a b"));

            Assert.Equal(VOCABULARY_SIZE, distribution.Length);
            Assert.Equal(1, distribution.Sum(p => (double)(float)p), 4);
        }

        [Fact]
        public void Generate_StopsAtTheEndToken()
        {
            ILanguageModel model = ScriptedModel(WordTokenizer.EOS);

            int[] generated = model.Generate(TOKENIZER.Encode("a"), new GenerationSettings { Temperature = 0, MaxNewTokens = 10 });

            Assert.Empty(generated);
        }

        [Fact]
        public void Generate_WithoutStoppingAtEndToken_ProducesMaxNewTokens()
        {
            ILanguageModel model = ScriptedModel(WordTokenizer.EOS);

            int[] generated = model.Generate(TOKENIZER.Encode("a"), new GenerationSettings { Temperature = 0, MaxNewTokens = 3, StopAtEndToken = false });

            Assert.Equal(new[] { WordTokenizer.EOS, WordTokenizer.EOS, WordTokenizer.EOS }, generated);
        }

        [Fact]
        public void Generate_ReturnsOnlyTheNewTokens()
        {
            ILanguageModel model = ScriptedModel(WordTokenizer.C);

            string continuation = model.Generate("a b", new GenerationSettings { Temperature = 0, MaxNewTokens = 3 });

            Assert.Equal("c c c", continuation);
        }

        [Fact]
        public void Generate_FeedsAtMostContextLengthTokens()
        {
            ScriptedSequenceNetwork network = new ScriptedSequenceNetwork(VOCABULARY_SIZE, MAX_SEQUENCE_LENGTH, WordTokenizer.B);
            ILanguageModel model = NetworkFactory.BuildLanguageModel(ModelSettings(contextLength: 3), TOKENIZER, network);

            model.Generate(TOKENIZER.Encode("a b c d"), new GenerationSettings { Temperature = 0, MaxNewTokens = 5 });

            Assert.Equal(5, network.PropagateCalls);
            Assert.Equal(3, network.LastSequenceLength);
        }

        [Fact]
        public void Generate_DefaultSettings_SampleUpToSixtyFourTokens()
        {
            ILanguageModel model = ScriptedModel(WordTokenizer.D);

            int[] generated = model.Generate(TOKENIZER.Encode("a"), null);

            Assert.Equal(64, generated.Length);
            Assert.All(generated, token => Assert.Equal(WordTokenizer.D, token));
        }

        [Fact]
        public void Generate_EmptyPromptWithoutBeginToken_Throws()
        {
            LanguageModelSettings settings = ModelSettings();
            settings.BeginToken = null;
            ILanguageModel model = NetworkFactory.BuildLanguageModel(settings, TOKENIZER, new ScriptedSequenceNetwork(VOCABULARY_SIZE, MAX_SEQUENCE_LENGTH, WordTokenizer.A));

            Assert.Throws<ArgumentException>(() => model.Generate(Array.Empty<int>(), new GenerationSettings()));
        }

        [Fact]
        public void Generate_EmptyPromptWithBeginToken_StartsFromIt()
        {
            ScriptedSequenceNetwork network = new ScriptedSequenceNetwork(VOCABULARY_SIZE, MAX_SEQUENCE_LENGTH, WordTokenizer.A);
            ILanguageModel model = NetworkFactory.BuildLanguageModel(ModelSettings(), TOKENIZER, network);

            int[] generated = model.Generate(Array.Empty<int>(), new GenerationSettings { Temperature = 0, MaxNewTokens = 2 });

            Assert.Equal(new[] { WordTokenizer.A, WordTokenizer.A }, generated);
            Assert.Equal(2, network.LastSequenceLength);
        }

        [Fact]
        public void Generate_InvalidTokensOrTemperature_Throw()
        {
            ILanguageModel model = ScriptedModel(WordTokenizer.A);

            Assert.Throws<ArgumentOutOfRangeException>(() => model.Generate(new[] { VOCABULARY_SIZE }, new GenerationSettings()));
            Assert.Throws<ArgumentOutOfRangeException>(() => model.Generate(new[] { WordTokenizer.A }, new GenerationSettings { Temperature = -1 }));
        }
    }
}
