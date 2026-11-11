using System;

namespace ANN.Net.Transformers
{
    /// <summary>
    /// Fixed sinusoidal positional encoding as described in "Attention Is All You Need"
    /// </summary>
    internal static class PositionalEncoding
    {
        private const double BASE = 10000;

        /// <summary>
        /// Builds the encoding table indexed as [position][dimension]
        /// </summary>
        public static Quad[][] Build(int maxLength, int dimension)
        {
            if (maxLength < 1) throw new ArgumentOutOfRangeException(nameof(maxLength), "Must be greater than 0");
            if (dimension < 1) throw new ArgumentOutOfRangeException(nameof(dimension), "Must be greater than 0");

            Quad[][] encodings = new Quad[maxLength][];
            for (int position = 0; position < maxLength; position++)
            {
                encodings[position] = new Quad[dimension];
                for (int i = 0; i < dimension; i++)
                {
                    double angle = position / Math.Pow(BASE, 2 * (i / 2) / (double)dimension);
                    encodings[position][i] = (Quad)(i % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle));
                }
            }

            return encodings;
        }
    }
}
