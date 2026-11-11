using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Settings;

namespace ANN.Net.Tokenizers
{
    /// <summary>
    /// Byte level byte pair encoding: the 256 byte values form the base vocabulary,
    /// training repeatedly merges the most frequent adjacent pair into a new token
    /// </summary>
    [Serializable]
    internal sealed class BpeTokenizer : ITokenizer
    {
        private const int BYTE_TOKENS_COUNT = 256;

        private readonly BpeTokenizerSettings settings;
        private readonly List<byte[]> vocabulary;
        private readonly Dictionary<string, int> specialTokens;
        private readonly Dictionary<(int Left, int Right), int> merges;

        [NonSerialized]
        private Dictionary<string, int[]> encodeCache;

        public BpeTokenizer(BpeTokenizerSettings settings)
        {
            if (settings == null) throw new ArgumentNullException(nameof(settings));
            Validate(settings);

            this.settings = settings;
            this.vocabulary = new List<byte[]>(settings.VocabularySize);
            this.specialTokens = new Dictionary<string, int>();
            this.merges = new Dictionary<(int Left, int Right), int>();

            for (int b = 0; b < BYTE_TOKENS_COUNT; b++)
            {
                this.vocabulary.Add(new[] { (byte)b });
            }

            foreach (string specialToken in settings.SpecialTokens)
            {
                this.specialTokens.Add(specialToken, this.vocabulary.Count);
                this.vocabulary.Add(Encoding.UTF8.GetBytes(specialToken));
            }
        }

        public int VocabularySize => this.vocabulary.Count;

        public IReadOnlyDictionary<string, int> SpecialTokens => this.specialTokens;

        /// <summary>
        /// Learned merges in the order they were learned; merge at index i produces token id 256 + specials count + i
        /// </summary>
        public IReadOnlyList<(int Left, int Right)> Merges => this.merges.OrderBy(m => m.Value).Select(m => m.Key).ToList();

        public void Train(IEnumerable<string> corpus)
        {
            if (corpus == null) throw new ArgumentNullException(nameof(corpus));

            this.encodeCache = null;

            // distinct chunks with their frequencies, each chunk as a mutable id sequence
            Dictionary<string, int> chunkFrequencies = new Dictionary<string, int>();
            foreach (string text in corpus)
            {
                if (text == null) continue;

                foreach (string chunk in TextChunker.Split(text))
                {
                    chunkFrequencies.TryGetValue(chunk, out int frequency);
                    chunkFrequencies[chunk] = frequency + 1;
                }
            }

            List<List<int>> chunks = new List<List<int>>(chunkFrequencies.Count);
            List<int> frequencies = new List<int>(chunkFrequencies.Count);
            foreach (KeyValuePair<string, int> entry in chunkFrequencies)
            {
                chunks.Add(this.ApplyMerges(ToByteIds(entry.Key)));
                frequencies.Add(entry.Value);
            }

            while (this.vocabulary.Count < this.settings.VocabularySize)
            {
                if (!TryFindBestPair(chunks, frequencies, out (int Left, int Right) pair, out int frequency)) break;
                if (frequency < this.settings.MinimumPairFrequency) break;

                int mergedId = this.vocabulary.Count;
                this.merges.Add(pair, mergedId);
                this.vocabulary.Add(this.vocabulary[pair.Left].Concat(this.vocabulary[pair.Right]).ToArray());

                foreach (List<int> chunk in chunks)
                {
                    MergePair(chunk, pair, mergedId);
                }
            }
        }

        public int[] Encode(string text)
        {
            if (text == null) throw new ArgumentNullException(nameof(text));

            if (this.encodeCache == null)
            {
                this.encodeCache = new Dictionary<string, int[]>();
            }

            List<int> tokens = new List<int>();
            foreach (string chunk in TextChunker.Split(text))
            {
                if (!this.encodeCache.TryGetValue(chunk, out int[] chunkTokens))
                {
                    chunkTokens = this.ApplyMerges(ToByteIds(chunk)).ToArray();
                    this.encodeCache[chunk] = chunkTokens;
                }

                tokens.AddRange(chunkTokens);
            }

            return tokens.ToArray();
        }

        public string Decode(IEnumerable<int> tokens)
        {
            if (tokens == null) throw new ArgumentNullException(nameof(tokens));

            List<byte> bytes = new List<byte>();
            foreach (int token in tokens)
            {
                if (token < 0 || token >= this.vocabulary.Count)
                    throw new ArgumentOutOfRangeException(nameof(tokens), $"Token id {token} is outside the vocabulary of size {this.vocabulary.Count}");

                bytes.AddRange(this.vocabulary[token]);
            }

            return Encoding.UTF8.GetString(bytes.ToArray());
        }

        /// <summary>
        /// Standard BPE encoding: keep merging the adjacent pair with the lowest rank until none applies
        /// </summary>
        private List<int> ApplyMerges(List<int> ids)
        {
            while (ids.Count > 1)
            {
                int bestRank = int.MaxValue;
                (int Left, int Right) bestPair = default;

                for (int i = 0; i < ids.Count - 1; i++)
                {
                    (int Left, int Right) pair = (ids[i], ids[i + 1]);
                    if (this.merges.TryGetValue(pair, out int rank) && rank < bestRank)
                    {
                        bestRank = rank;
                        bestPair = pair;
                    }
                }

                if (bestRank == int.MaxValue) break;

                MergePair(ids, bestPair, bestRank);
            }

            return ids;
        }

        private static bool TryFindBestPair(List<List<int>> chunks, List<int> frequencies, out (int Left, int Right) bestPair, out int bestFrequency)
        {
            Dictionary<(int Left, int Right), int> pairFrequencies = new Dictionary<(int Left, int Right), int>();
            for (int c = 0; c < chunks.Count; c++)
            {
                List<int> chunk = chunks[c];
                for (int i = 0; i < chunk.Count - 1; i++)
                {
                    (int Left, int Right) pair = (chunk[i], chunk[i + 1]);
                    pairFrequencies.TryGetValue(pair, out int frequency);
                    pairFrequencies[pair] = frequency + frequencies[c];
                }
            }

            bestPair = default;
            bestFrequency = 0;
            foreach (KeyValuePair<(int Left, int Right), int> entry in pairFrequencies)
            {
                // ties break on the lowest pair so training is deterministic
                if (entry.Value > bestFrequency
                    || (entry.Value == bestFrequency && entry.Key.CompareTo(bestPair) < 0))
                {
                    bestFrequency = entry.Value;
                    bestPair = entry.Key;
                }
            }

            return bestFrequency > 0;
        }

        private static void MergePair(List<int> ids, (int Left, int Right) pair, int mergedId)
        {
            int write = 0;
            for (int read = 0; read < ids.Count; read++)
            {
                if (read < ids.Count - 1 && ids[read] == pair.Left && ids[read + 1] == pair.Right)
                {
                    ids[write++] = mergedId;
                    read++;
                }
                else
                {
                    ids[write++] = ids[read];
                }
            }

            ids.RemoveRange(write, ids.Count - write);
        }

        private static List<int> ToByteIds(string chunk)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(chunk);
            List<int> ids = new List<int>(bytes.Length);
            foreach (byte b in bytes)
            {
                ids.Add(b);
            }

            return ids;
        }

        private static void Validate(BpeTokenizerSettings settings)
        {
            if (settings.SpecialTokens == null) throw new ArgumentNullException(nameof(settings.SpecialTokens));

            int specialTokensCount = settings.SpecialTokens.Count;
            if (settings.VocabularySize < BYTE_TOKENS_COUNT + specialTokensCount)
                throw new ArgumentException($"Must be at least {BYTE_TOKENS_COUNT + specialTokensCount} (256 bytes + {specialTokensCount} special tokens)", nameof(settings.VocabularySize));

            if (settings.SpecialTokens.Any(string.IsNullOrEmpty))
                throw new ArgumentException("Must not contain empty names", nameof(settings.SpecialTokens));

            if (settings.SpecialTokens.Distinct().Count() != specialTokensCount)
                throw new ArgumentException("Must not contain duplicate names", nameof(settings.SpecialTokens));
        }
    }
}
