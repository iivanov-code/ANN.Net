using System;
using System.Collections.Generic;

namespace ANN.Net.Tokenizers
{
    /// <summary>
    /// Splits text into the pieces inside which byte pair merges are allowed:
    /// a run of non-whitespace characters together with the whitespace that precedes it
    /// </summary>
    internal static class TextChunker
    {
        public static IList<string> Split(string text)
        {
            if (text == null) throw new ArgumentNullException(nameof(text));

            List<string> chunks = new List<string>();
            int start = 0;
            int position = 0;

            while (position < text.Length)
            {
                // the whitespace belongs to the word that follows it
                while (position < text.Length && char.IsWhiteSpace(text[position]))
                {
                    position++;
                }

                while (position < text.Length && !char.IsWhiteSpace(text[position]))
                {
                    position++;
                }

                chunks.Add(text.Substring(start, position - start));
                start = position;
            }

            return chunks;
        }
    }
}
