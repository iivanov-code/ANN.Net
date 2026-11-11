using System.Collections.Generic;

namespace ANN.Net.Abstractions.Interfaces
{
    /// <summary>
    /// Turns text into a sequence of token ids and back
    /// </summary>
    public interface ITokenizer
    {
        /// <summary>
        /// Number of distinct token ids the tokenizer can currently produce; ids are in [0, VocabularySize)
        /// </summary>
        int VocabularySize { get; }

        /// <summary>
        /// Reserved tokens (padding, begin, end, ...) by name. They are never produced from plain text
        /// </summary>
        IReadOnlyDictionary<string, int> SpecialTokens { get; }

        /// <summary>
        /// Learns the vocabulary from the given texts
        /// </summary>
        void Train(IEnumerable<string> corpus);

        int[] Encode(string text);

        string Decode(IEnumerable<int> tokens);
    }
}
