using System;

namespace ANN.Net.Embeddings
{
    /// <summary>
    /// Unigram distribution raised to a power, sampled by binary search over the cumulative weights
    /// </summary>
    [Serializable]
    internal sealed class NoiseDistribution
    {
        private readonly double[] cumulativeWeights;

        public NoiseDistribution(long[] counts, double power)
        {
            if (counts == null) throw new ArgumentNullException(nameof(counts));
            if (power < 0) throw new ArgumentOutOfRangeException(nameof(power), "Must not be negative");

            this.cumulativeWeights = new double[counts.Length];

            double total = 0;
            for (int token = 0; token < counts.Length; token++)
            {
                if (counts[token] > 0)
                {
                    total += Math.Pow(counts[token], power);
                }

                this.cumulativeWeights[token] = total;
            }
        }

        public double Total => this.cumulativeWeights.Length == 0 ? 0 : this.cumulativeWeights[this.cumulativeWeights.Length - 1];

        public bool HasSamples => this.Total > 0;

        /// <summary>
        /// Draws a token; tokens with a count of zero are never drawn
        /// </summary>
        public int Sample(Random random)
        {
            if (random == null) throw new ArgumentNullException(nameof(random));
            if (!this.HasSamples) throw new InvalidOperationException("The distribution has no tokens with a positive count");

            double point = random.NextDouble() * this.Total;

            // first token whose cumulative weight exceeds the point; zero-width tokens are skipped
            int low = 0;
            int high = this.cumulativeWeights.Length - 1;
            while (low < high)
            {
                int middle = low + (high - low) / 2;
                if (this.cumulativeWeights[middle] > point)
                {
                    high = middle;
                }
                else
                {
                    low = middle + 1;
                }
            }

            return low;
        }
    }
}
