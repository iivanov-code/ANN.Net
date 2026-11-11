using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;

namespace ANN.Net.Tests.LanguageModels
{
    /// <summary>
    /// Fixed vocabulary of whole words separated by spaces; ids 0 and 1 are the specials
    /// </summary>
    internal sealed class WordTokenizer : ITokenizer
    {
        public const int BOS = 0;
        public const int EOS = 1;
        public const int A = 2;
        public const int B = 3;
        public const int C = 4;
        public const int D = 5;

        private static readonly string[] WORDS = { "<bos>", "<eos>", "a", "b", "c", "d" };

        public int VocabularySize => WORDS.Length;

        public IReadOnlyDictionary<string, int> SpecialTokens { get; } = new Dictionary<string, int>
        {
            ["<bos>"] = BOS,
            ["<eos>"] = EOS
        };

        public void Train(IEnumerable<string> corpus)
        {
        }

        public int[] Encode(string text)
        {
            return text.Split(' ', StringSplitOptions.RemoveEmptyEntries)
                       .Select(word => Array.IndexOf(WORDS, word))
                       .ToArray();
        }

        public string Decode(IEnumerable<int> tokens)
        {
            return string.Join(" ", tokens.Select(t => WORDS[t]));
        }
    }

    /// <summary>
    /// Sequence network that always predicts the same next token with the given probability
    /// and remembers how long the last propagated sequence was
    /// </summary>
    internal sealed class ScriptedSequenceNetwork : ISequenceNetwork
    {
        private readonly int nextToken;
        private readonly float probability;

        public ScriptedSequenceNetwork(int dimension, int maxSequenceLength, int nextToken, float probability = 1f)
        {
            this.InputDimension = dimension;
            this.OutputDimension = dimension;
            this.MaxSequenceLength = maxSequenceLength;
            this.nextToken = nextToken;
            this.probability = probability;
        }

        public int LastSequenceLength { get; private set; }

        public int PropagateCalls { get; private set; }

        public Quad[][] Values { get; private set; }

        public Quad[][] Errors { get; private set; }

        public ICollection<LearningRateOptimizer> Optimizers { get; } = new List<LearningRateOptimizer>();

        public int InputDimension { get; }

        public int OutputDimension { get; }

        public int MaxSequenceLength { get; }

        public bool OutputApplySoftMax { get; set; } = true;

        public Quad[][] Propagate(Quad[][] sequence)
        {
            if (sequence.Length > this.MaxSequenceLength) throw new ArgumentException("Too long", nameof(sequence));

            this.LastSequenceLength = sequence.Length;
            this.PropagateCalls++;

            Quad[][] outputs = new Quad[sequence.Length][];
            float rest = (1 - this.probability) / (this.OutputDimension - 1);
            for (int t = 0; t < sequence.Length; t++)
            {
                outputs[t] = new Quad[this.OutputDimension];
                for (int i = 0; i < this.OutputDimension; i++)
                {
                    outputs[t][i] = i == this.nextToken ? this.probability : rest;
                }
            }

            this.Values = outputs;
            return outputs;
        }

        public Quad Backpropagate(Quad[][] targets)
        {
            this.Errors = targets;
            return 0;
        }
    }
}
