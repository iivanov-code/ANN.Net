using System;
using System.Collections.Generic;

namespace ANN.Net.LanguageModels
{
    /// <summary>
    /// Cuts token sequences into next-token training windows
    /// </summary>
    internal static class TokenWindows
    {
        /// <summary>
        /// Wraps every sequence in the begin and end tokens and cuts it into windows of
        /// contextLength + 1 tokens starting every stride tokens. The inputs of a window are
        /// all but its last token, the targets all but its first. Leftovers shorter than two tokens are dropped
        /// </summary>
        public static List<int[]> Build(IEnumerable<int[]> sequences, int contextLength, int stride, int? beginToken, int? endToken)
        {
            if (sequences == null) throw new ArgumentNullException(nameof(sequences));
            if (contextLength < 1) throw new ArgumentOutOfRangeException(nameof(contextLength), "Must be greater than 0");
            if (stride < 1) throw new ArgumentOutOfRangeException(nameof(stride), "Must be greater than 0");

            int windowLength = contextLength + 1;
            List<int[]> windows = new List<int[]>();

            foreach (int[] sequence in sequences)
            {
                if (sequence == null) throw new ArgumentException("Sequences must not contain null", nameof(sequences));

                int[] tokens = Wrap(sequence, beginToken, endToken);
                if (tokens.Length < 2) continue;

                for (int start = 0; start < tokens.Length; start += stride)
                {
                    int end = Math.Min(start + windowLength, tokens.Length);
                    if (end - start >= 2)
                    {
                        int[] window = new int[end - start];
                        Array.Copy(tokens, start, window, 0, window.Length);
                        windows.Add(window);
                    }

                    if (end == tokens.Length) break;
                }
            }

            return windows;
        }

        private static int[] Wrap(int[] sequence, int? beginToken, int? endToken)
        {
            int length = sequence.Length + (beginToken.HasValue ? 1 : 0) + (endToken.HasValue ? 1 : 0);
            int[] tokens = new int[length];
            int offset = 0;

            if (beginToken.HasValue)
            {
                tokens[offset++] = beginToken.Value;
            }

            Array.Copy(sequence, 0, tokens, offset, sequence.Length);
            offset += sequence.Length;

            if (endToken.HasValue)
            {
                tokens[offset] = endToken.Value;
            }

            return tokens;
        }
    }
}
