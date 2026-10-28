using System.Collections.Generic;
using System.ComponentModel;

namespace ANN.Net.Abstractions.Settings
{
    public class BpeTokenizerSettings
    {
        /// <summary>
        /// Target vocabulary size. Must hold the 256 byte tokens plus the special tokens
        /// </summary>
        [DefaultValue(1024)]
        public ushort VocabularySize { get; set; } = 1024;

        /// <summary>
        /// Training stops when the most frequent adjacent pair occurs fewer times than this
        /// </summary>
        [DefaultValue(2)]
        public ushort MinimumPairFrequency { get; set; } = 2;

        /// <summary>
        /// Reserved token names, assigned the ids right after the 256 byte tokens, in order
        /// </summary>
        public IList<string> SpecialTokens { get; set; } = new List<string>();
    }
}
