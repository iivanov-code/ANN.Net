using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Utils;

namespace ANN.Net.LanguageModels
{
    /// <summary>
    /// Next-token language model: a tokenizer, optional token embeddings and a sequence network
    /// whose outputs are probability distributions over the vocabulary
    /// </summary>
    [Serializable]
    internal sealed class LanguageModel : ILanguageModel
    {
        private const double LOG_EPSILON = 1e-7;

        private readonly LanguageModelSettings settings;
        private readonly int? beginToken;
        private readonly int? endToken;
        private readonly int stride;

        [NonSerialized]
        private Random random;

        public LanguageModel(LanguageModelSettings settings, ITokenizer tokenizer, ISequenceNetwork network, IEmbeddingNetwork embeddings)
        {
            if (settings == null) throw new ArgumentNullException(nameof(settings));
            if (tokenizer == null) throw new ArgumentNullException(nameof(tokenizer));
            if (network == null) throw new ArgumentNullException(nameof(network));

            Validate(settings, tokenizer, network, embeddings);

            this.settings = settings;
            this.Tokenizer = tokenizer;
            this.Network = network;
            this.Embeddings = embeddings;
            this.beginToken = ResolveSpecialToken(tokenizer, settings.BeginToken, nameof(settings.BeginToken));
            this.endToken = ResolveSpecialToken(tokenizer, settings.EndToken, nameof(settings.EndToken));
            this.stride = settings.WindowStride == 0 ? settings.ContextLength : settings.WindowStride;
        }

        public ITokenizer Tokenizer { get; }

        public ISequenceNetwork Network { get; }

        public IEmbeddingNetwork Embeddings { get; }

        public int ContextLength => this.settings.ContextLength;

        private Random Random => this.random ?? (this.random = CreateRandom(this.settings.Seed));

        public Quad Train(IEnumerable<int[]> sequences)
        {
            List<int[]> windows = this.BuildWindows(sequences);
            if (this.settings.ShuffleWindows)
            {
                this.Shuffle(windows);
            }

            double totalLoss = 0;
            long predictedCount = 0;

            foreach (int[] window in windows)
            {
                int length = window.Length - 1;
                Quad[][] inputs = this.Encode(window, 0, length);
                Quad[][] targets = new Quad[length][];
                for (int t = 0; t < length; t++)
                {
                    targets[t] = new Quad[this.Tokenizer.VocabularySize];
                    targets[t][window[t + 1]] = 1;
                }

                this.Network.Propagate(inputs);
                totalLoss += this.Network.Backpropagate(targets);
                predictedCount += length;
            }

            return predictedCount == 0 ? 0 : (Quad)(totalLoss / predictedCount);
        }

        public Quad Evaluate(IEnumerable<int[]> sequences)
        {
            List<int[]> windows = this.BuildWindows(sequences);

            double totalLoss = 0;
            long predictedCount = 0;

            foreach (int[] window in windows)
            {
                int length = window.Length - 1;
                Quad[][] outputs = this.Network.Propagate(this.Encode(window, 0, length));
                for (int t = 0; t < length; t++)
                {
                    double probability = Math.Max((double)(float)outputs[t][window[t + 1]], LOG_EPSILON);
                    totalLoss -= Math.Log(probability);
                }

                predictedCount += length;
            }

            return predictedCount == 0 ? 0 : (Quad)(totalLoss / predictedCount);
        }

        public Quad Perplexity(IEnumerable<int[]> sequences)
        {
            return (Quad)Math.Exp(this.Evaluate(sequences));
        }

        public Quad[] PredictNext(int[] prompt)
        {
            if (prompt == null) throw new ArgumentNullException(nameof(prompt));
            this.ValidateTokens(prompt, nameof(prompt));

            List<int> sequence = this.StartSequence(prompt);
            return (Quad[])this.PredictNext(sequence).Clone();
        }

        public int[] Generate(int[] prompt, GenerationSettings settings)
        {
            if (prompt == null) throw new ArgumentNullException(nameof(prompt));
            this.ValidateTokens(prompt, nameof(prompt));
            if (settings == null)
            {
                settings = new GenerationSettings();
            }

            if (float.IsNaN(settings.Temperature) || settings.Temperature < 0)
                throw new ArgumentOutOfRangeException(nameof(settings), "Temperature must not be negative");

            List<int> sequence = this.StartSequence(prompt);
            List<int> generated = new List<int>(settings.MaxNewTokens);

            for (int i = 0; i < settings.MaxNewTokens; i++)
            {
                Quad[] distribution = this.PredictNext(sequence);
                int next = TokenSampler.Sample(distribution, settings.Temperature, settings.TopK, this.Random);

                if (settings.StopAtEndToken && this.endToken.HasValue && next == this.endToken.Value)
                {
                    break;
                }

                sequence.Add(next);
                generated.Add(next);
            }

            return generated.ToArray();
        }

        public string Generate(string prompt, GenerationSettings settings)
        {
            if (prompt == null) throw new ArgumentNullException(nameof(prompt));

            int[] tokens = this.Generate(this.Tokenizer.Encode(prompt), settings);
            return this.Tokenizer.Decode(tokens);
        }

        /// <summary>
        /// Prompt with the begin token in front, unless it already starts with it
        /// </summary>
        private List<int> StartSequence(int[] prompt)
        {
            List<int> sequence = new List<int>(prompt.Length + 1);
            if (this.beginToken.HasValue && (prompt.Length == 0 || prompt[0] != this.beginToken.Value))
            {
                sequence.Add(this.beginToken.Value);
            }

            sequence.AddRange(prompt);

            if (sequence.Count == 0)
                throw new ArgumentException("The prompt must not be empty when no begin token is configured", nameof(prompt));

            return sequence;
        }

        /// <summary>
        /// Distribution of the last position after propagating the last ContextLength tokens of the sequence
        /// </summary>
        private Quad[] PredictNext(List<int> sequence)
        {
            int length = Math.Min(sequence.Count, this.ContextLength);
            int start = sequence.Count - length;

            Quad[][] outputs = this.Network.Propagate(this.Encode(sequence, start, length));
            return outputs[outputs.Length - 1];
        }

        private List<int[]> BuildWindows(IEnumerable<int[]> sequences)
        {
            if (sequences == null) throw new ArgumentNullException(nameof(sequences));

            List<int[]> windows = TokenWindows.Build(sequences, this.ContextLength, this.stride, this.beginToken, this.endToken);
            foreach (int[] window in windows)
            {
                this.ValidateTokens(window, nameof(sequences));
            }

            return windows;
        }

        private Quad[][] Encode(IReadOnlyList<int> tokens, int start, int length)
        {
            int[] slice = new int[length];
            for (int t = 0; t < length; t++)
            {
                slice[t] = tokens[start + t];
            }

            if (this.Embeddings != null)
            {
                return this.Embeddings.Embed(slice);
            }

            int vocabularySize = this.Tokenizer.VocabularySize;
            Quad[][] vectors = new Quad[length][];
            for (int t = 0; t < length; t++)
            {
                vectors[t] = new Quad[vocabularySize];
                vectors[t][slice[t]] = 1;
            }

            return vectors;
        }

        private void Shuffle(List<int[]> windows)
        {
            for (int i = windows.Count - 1; i > 0; i--)
            {
                int j = this.Random.Next(i + 1);
                int[] swap = windows[i];
                windows[i] = windows[j];
                windows[j] = swap;
            }
        }

        private void ValidateTokens(IEnumerable<int> tokens, string parameterName)
        {
            int vocabularySize = this.Tokenizer.VocabularySize;
            foreach (int token in tokens)
            {
                if (token < 0 || token >= vocabularySize)
                    throw new ArgumentOutOfRangeException(parameterName, $"Token {token} is outside the vocabulary of size {vocabularySize}");
            }
        }

        private static int? ResolveSpecialToken(ITokenizer tokenizer, string name, string parameterName)
        {
            if (string.IsNullOrEmpty(name))
            {
                return null;
            }

            if (tokenizer.SpecialTokens == null || !tokenizer.SpecialTokens.TryGetValue(name, out int token))
                throw new ArgumentException($"The tokenizer has no special token named '{name}'", parameterName);

            return token;
        }

        private static Random CreateRandom(int? seed)
        {
            return seed.HasValue ? new Random(seed.Value) : new Random(NetworkUtils.Rand.Next());
        }

        private static void Validate(LanguageModelSettings settings, ITokenizer tokenizer, ISequenceNetwork network, IEmbeddingNetwork embeddings)
        {
            if (settings.ContextLength < 1)
                throw new ArgumentException("Must be greater than 0", nameof(settings.ContextLength));
            if (settings.ContextLength > network.MaxSequenceLength)
                throw new ArgumentException($"Must not exceed the network's maximum sequence length of {network.MaxSequenceLength}", nameof(settings.ContextLength));
            if (tokenizer.VocabularySize < 2)
                throw new ArgumentException("The tokenizer must be trained before the model is built", nameof(tokenizer));
            if (!network.OutputApplySoftMax)
                throw new ArgumentException("The network must apply soft-max to its outputs", nameof(network));
            if (network.OutputDimension != tokenizer.VocabularySize)
                throw new ArgumentException($"The network output dimension {network.OutputDimension} differs from the vocabulary size {tokenizer.VocabularySize}", nameof(network));

            if (embeddings == null)
            {
                if (network.InputDimension != tokenizer.VocabularySize)
                    throw new ArgumentException($"Without embeddings the network input dimension {network.InputDimension} must equal the vocabulary size {tokenizer.VocabularySize}", nameof(network));
            }
            else
            {
                if (embeddings.VocabularySize != tokenizer.VocabularySize)
                    throw new ArgumentException($"The embeddings vocabulary size {embeddings.VocabularySize} differs from the tokenizer's {tokenizer.VocabularySize}", nameof(embeddings));
                if (network.InputDimension != embeddings.EmbeddingDimension)
                    throw new ArgumentException($"The network input dimension {network.InputDimension} differs from the embedding dimension {embeddings.EmbeddingDimension}", nameof(network));
            }
        }
    }
}
