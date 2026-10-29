using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Transformers;
using ANN.Net.Utils;

namespace ANN.Net.Embeddings
{
    /// <summary>
    /// Word2vec with negative sampling: an input and an output embedding matrix,
    /// trained with skip-gram or continuous bag of words samples.
    /// Every weight owns its optimizer and is updated as soon as its gradient is known
    /// </summary>
    [Serializable]
    internal sealed class Word2VecNetwork : IEmbeddingNetwork
    {
        private const double LOG_EPSILON = 1e-7;
        private const int NEGATIVE_REDRAW_ATTEMPTS = 10;

        private readonly Word2VecSettings settings;
        private readonly Quad[][] inputEmbeddings;
        private readonly Quad[][] outputEmbeddings;
        private readonly IOptimizer[][] inputOptimizers;
        private readonly IOptimizer[][] outputOptimizers;
        private readonly Quad[] sampleTargets;

        [NonSerialized]
        private Random random;

        private int[] lastInputTokens;
        private int[] lastOutputTokens;
        private Quad[] lastHidden;

        public Word2VecNetwork(Word2VecSettings settings)
        {
            if (settings == null) throw new ArgumentNullException(nameof(settings));
            Validate(settings);

            this.settings = settings;
            this.Optimizers = new List<LearningRateOptimizer>();
            this.random = CreateRandom(settings.Seed);

            int vocabularySize = settings.VocabularySize;
            int dimension = settings.EmbeddingDimension;

            this.inputEmbeddings = new Quad[vocabularySize][];
            this.outputEmbeddings = new Quad[vocabularySize][];
            this.inputOptimizers = new IOptimizer[vocabularySize][];
            this.outputOptimizers = new IOptimizer[vocabularySize][];

            // as in the reference implementation: small uniform input vectors, zero output vectors
            double limit = 0.5 / dimension;
            for (int token = 0; token < vocabularySize; token++)
            {
                this.inputEmbeddings[token] = new Quad[dimension];
                this.outputEmbeddings[token] = new Quad[dimension];
                this.inputOptimizers[token] = new IOptimizer[dimension];
                this.outputOptimizers[token] = new IOptimizer[dimension];

                for (int d = 0; d < dimension; d++)
                {
                    this.inputEmbeddings[token][d] = (Quad)(this.random.NextDouble() * 2 * limit - limit);
                    this.outputEmbeddings[token][d] = 0;
                    this.inputOptimizers[token][d] = TransformerUtils.CreateOptimizer(settings.OptimizerType, this.Optimizers);
                    this.outputOptimizers[token][d] = TransformerUtils.CreateOptimizer(settings.OptimizerType, this.Optimizers);
                }
            }

            // the first output token of a training sample is the real context, the rest are noise
            this.sampleTargets = new Quad[1 + settings.NegativeSamplesCount];
            this.sampleTargets[0] = 1;
        }

        public int VocabularySize => this.settings.VocabularySize;

        public int EmbeddingDimension => this.settings.EmbeddingDimension;

        public Quad[][] Embeddings => this.inputEmbeddings;

        public Quad[] Values { get; private set; }

        public Quad[] Errors { get; private set; }

        public ICollection<LearningRateOptimizer> Optimizers { get; }

        private Random Random => this.random ?? (this.random = CreateRandom(this.settings.Seed));

        public Quad[] Propagate(int[] inputTokens, int[] outputTokens)
        {
            if (inputTokens == null) throw new ArgumentNullException(nameof(inputTokens));
            if (outputTokens == null) throw new ArgumentNullException(nameof(outputTokens));
            if (inputTokens.Length < 1) throw new ArgumentException("At least one input token is required", nameof(inputTokens));
            if (outputTokens.Length < 1) throw new ArgumentException("At least one output token is required", nameof(outputTokens));
            this.ValidateTokens(inputTokens, nameof(inputTokens));
            this.ValidateTokens(outputTokens, nameof(outputTokens));

            int dimension = this.EmbeddingDimension;

            Quad[] hidden = new Quad[dimension];
            foreach (int token in inputTokens)
            {
                Quad[] embedding = this.inputEmbeddings[token];
                for (int d = 0; d < dimension; d++)
                {
                    hidden[d] += embedding[d];
                }
            }

            if (inputTokens.Length > 1)
            {
                for (int d = 0; d < dimension; d++)
                {
                    hidden[d] /= inputTokens.Length;
                }
            }

            Quad[] scores = new Quad[outputTokens.Length];
            for (int k = 0; k < outputTokens.Length; k++)
            {
                Quad[] embedding = this.outputEmbeddings[outputTokens[k]];
                double logit = 0;
                for (int d = 0; d < dimension; d++)
                {
                    logit += (double)(float)(hidden[d] * embedding[d]);
                }

                scores[k] = Sigmoid(logit);
            }

            this.lastInputTokens = inputTokens;
            this.lastOutputTokens = outputTokens;
            this.lastHidden = hidden;
            this.Values = scores;
            return scores;
        }

        public Quad Backpropagate(Quad[] targets)
        {
            if (targets == null) throw new ArgumentNullException(nameof(targets));
            if (this.Values == null) throw new InvalidOperationException("Propagate must be called before Backpropagate");
            if (targets.Length != this.lastOutputTokens.Length)
                throw new ArgumentException("Targets count differs from the propagated output tokens count", nameof(targets));

            int dimension = this.EmbeddingDimension;

            Quad totalLoss = 0;
            Quad[] errors = new Quad[targets.Length];
            Quad[] hiddenGradient = new Quad[dimension];

            // the gradient towards the hidden vector uses the output embeddings before they are updated
            for (int k = 0; k < targets.Length; k++)
            {
                Quad score = this.Values[k];
                Quad target = targets[k];

                errors[k] = score - target;
                totalLoss += CalculateLoss(target, score);

                Quad[] embedding = this.outputEmbeddings[this.lastOutputTokens[k]];
                for (int d = 0; d < dimension; d++)
                {
                    hiddenGradient[d] += errors[k] * embedding[d];
                }
            }

            this.Errors = errors;

            for (int k = 0; k < targets.Length; k++)
            {
                int token = this.lastOutputTokens[k];
                Quad[] embedding = this.outputEmbeddings[token];
                IOptimizer[] optimizers = this.outputOptimizers[token];
                for (int d = 0; d < dimension; d++)
                {
                    embedding[d] = optimizers[d].Calculate(errors[k] * this.lastHidden[d], embedding[d]);
                }
            }

            // the hidden vector is the mean of the input embeddings, so each one gets an equal share
            Quad share = (Quad)(1.0 / this.lastInputTokens.Length);
            foreach (int token in this.lastInputTokens)
            {
                Quad[] embedding = this.inputEmbeddings[token];
                IOptimizer[] optimizers = this.inputOptimizers[token];
                for (int d = 0; d < dimension; d++)
                {
                    embedding[d] = optimizers[d].Calculate(hiddenGradient[d] * share, embedding[d]);
                }
            }

            return totalLoss;
        }

        public Quad Train(IEnumerable<int[]> sequences)
        {
            if (sequences == null) throw new ArgumentNullException(nameof(sequences));

            List<int[]> corpus = sequences.ToList();

            long[] counts = new long[this.VocabularySize];
            long totalCount = 0;
            foreach (int[] sequence in corpus)
            {
                if (sequence == null) throw new ArgumentException("Sequences must not contain null", nameof(sequences));
                this.ValidateTokens(sequence, nameof(sequences));

                foreach (int token in sequence)
                {
                    counts[token]++;
                    totalCount++;
                }
            }

            if (totalCount == 0)
            {
                return 0;
            }

            NoiseDistribution noise = new NoiseDistribution(counts, this.settings.NoiseDistributionPower);

            double totalLoss = 0;
            long samplesCount = 0;

            foreach (int[] sequence in corpus)
            {
                int[] tokens = this.Subsample(sequence, counts, totalCount);

                for (int center = 0; center < tokens.Length; center++)
                {
                    int window = this.settings.UseDynamicWindow
                        ? this.Random.Next(1, this.settings.WindowSize + 1)
                        : this.settings.WindowSize;

                    int start = Math.Max(0, center - window);
                    int end = Math.Min(tokens.Length - 1, center + window);

                    if (this.settings.Architecture == Word2VecArchitecture.SkipGram)
                    {
                        for (int position = start; position <= end; position++)
                        {
                            if (position == center) continue;

                            totalLoss += this.TrainSample(new[] { tokens[center] }, tokens[position], noise);
                            samplesCount++;
                        }
                    }
                    else
                    {
                        int contextLength = end - start;
                        if (contextLength == 0) continue;

                        int[] context = new int[contextLength];
                        int index = 0;
                        for (int position = start; position <= end; position++)
                        {
                            if (position == center) continue;
                            context[index++] = tokens[position];
                        }

                        totalLoss += this.TrainSample(context, tokens[center], noise);
                        samplesCount++;
                    }
                }
            }

            return samplesCount == 0 ? 0 : (Quad)(totalLoss / samplesCount);
        }

        public Quad[] GetEmbedding(int token)
        {
            this.ValidateToken(token, nameof(token));
            return (Quad[])this.inputEmbeddings[token].Clone();
        }

        public Quad[][] Embed(int[] tokens)
        {
            if (tokens == null) throw new ArgumentNullException(nameof(tokens));
            this.ValidateTokens(tokens, nameof(tokens));

            Quad[][] vectors = new Quad[tokens.Length][];
            for (int t = 0; t < tokens.Length; t++)
            {
                vectors[t] = (Quad[])this.inputEmbeddings[tokens[t]].Clone();
            }

            return vectors;
        }

        public Quad CosineSimilarity(int first, int second)
        {
            this.ValidateToken(first, nameof(first));
            this.ValidateToken(second, nameof(second));

            return CosineSimilarity(this.inputEmbeddings[first], this.inputEmbeddings[second]);
        }

        public IReadOnlyList<(int Token, Quad Similarity)> MostSimilar(int token, int count)
        {
            this.ValidateToken(token, nameof(token));
            if (count < 1) throw new ArgumentOutOfRangeException(nameof(count), "Must be greater than 0");

            Quad[] embedding = this.inputEmbeddings[token];
            List<(int Token, Quad Similarity)> similarities = new List<(int Token, Quad Similarity)>(this.VocabularySize - 1);
            for (int other = 0; other < this.VocabularySize; other++)
            {
                if (other == token) continue;
                similarities.Add((other, CosineSimilarity(embedding, this.inputEmbeddings[other])));
            }

            return similarities.OrderByDescending(s => s.Similarity)
                               .ThenBy(s => s.Token)
                               .Take(count)
                               .ToList();
        }

        private double TrainSample(int[] inputTokens, int positiveToken, NoiseDistribution noise)
        {
            int[] outputTokens = new int[this.sampleTargets.Length];
            outputTokens[0] = positiveToken;
            for (int k = 1; k < outputTokens.Length; k++)
            {
                outputTokens[k] = this.SampleNegative(noise, positiveToken);
            }

            this.Propagate(inputTokens, outputTokens);
            return this.Backpropagate(this.sampleTargets);
        }

        private int SampleNegative(NoiseDistribution noise, int positiveToken)
        {
            int negative = noise.Sample(this.Random);
            for (int attempt = 1; attempt < NEGATIVE_REDRAW_ATTEMPTS && negative == positiveToken; attempt++)
            {
                negative = noise.Sample(this.Random);
            }

            return negative;
        }

        /// <summary>
        /// Drops frequent tokens with the keep probability (sqrt(f / t) + 1) * t / f of the reference implementation
        /// </summary>
        private int[] Subsample(int[] sequence, long[] counts, long totalCount)
        {
            double threshold = this.settings.SubsamplingThreshold;
            if (threshold <= 0)
            {
                return sequence;
            }

            List<int> kept = new List<int>(sequence.Length);
            foreach (int token in sequence)
            {
                double frequency = (double)counts[token] / totalCount;
                double keepProbability = (Math.Sqrt(frequency / threshold) + 1) * threshold / frequency;
                if (keepProbability >= 1 || this.Random.NextDouble() < keepProbability)
                {
                    kept.Add(token);
                }
            }

            return kept.Count == sequence.Length ? sequence : kept.ToArray();
        }

        private void ValidateTokens(int[] tokens, string parameterName)
        {
            foreach (int token in tokens)
            {
                this.ValidateToken(token, parameterName);
            }
        }

        private void ValidateToken(int token, string parameterName)
        {
            if (token < 0 || token >= this.VocabularySize)
                throw new ArgumentOutOfRangeException(parameterName, $"Token {token} is outside the vocabulary of size {this.VocabularySize}");
        }

        private static Random CreateRandom(int? seed)
        {
            return seed.HasValue ? new Random(seed.Value) : new Random(NetworkUtils.Rand.Next());
        }

        private static Quad Sigmoid(double logit)
        {
            return (Quad)(1.0 / (1.0 + Math.Exp(-logit)));
        }

        private static Quad CalculateLoss(Quad target, Quad score)
        {
            double probability = Math.Min(Math.Max((double)(float)score, LOG_EPSILON), 1 - LOG_EPSILON);
            return (Quad)(-(target * Math.Log(probability) + (1 - target) * Math.Log(1 - probability)));
        }

        private static Quad CosineSimilarity(Quad[] first, Quad[] second)
        {
            double dot = 0;
            double firstNorm = 0;
            double secondNorm = 0;
            for (int d = 0; d < first.Length; d++)
            {
                double a = first[d];
                double b = second[d];
                dot += a * b;
                firstNorm += a * a;
                secondNorm += b * b;
            }

            if (firstNorm == 0 || secondNorm == 0)
            {
                return 0;
            }

            return (Quad)(dot / Math.Sqrt(firstNorm * secondNorm));
        }

        private static void Validate(Word2VecSettings settings)
        {
            if (settings.VocabularySize < 2) throw new ArgumentException("Must be greater than 1", nameof(settings.VocabularySize));
            if (settings.EmbeddingDimension < 1) throw new ArgumentException("Must be greater than 0", nameof(settings.EmbeddingDimension));
            if (settings.WindowSize < 1) throw new ArgumentException("Must be greater than 0", nameof(settings.WindowSize));
            if (settings.NegativeSamplesCount < 1) throw new ArgumentException("Must be greater than 0", nameof(settings.NegativeSamplesCount));
            if (settings.NoiseDistributionPower < 0) throw new ArgumentException("Must not be negative", nameof(settings.NoiseDistributionPower));
            if (settings.SubsamplingThreshold < 0) throw new ArgumentException("Must not be negative", nameof(settings.SubsamplingThreshold));
            if (float.IsNaN(settings.NoiseDistributionPower) || float.IsNaN(settings.SubsamplingThreshold))
                throw new ArgumentException("Must be a number", nameof(settings));
        }
    }
}
