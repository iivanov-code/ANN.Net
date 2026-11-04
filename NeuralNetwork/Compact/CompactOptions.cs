using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;

namespace ANN.Net
{
    /// <summary>
    /// How <see cref="NetworkCompactor"/> stores the weights of a compact network
    /// </summary>
    public sealed class CompactOptions
    {
        /// <summary>
        /// Number format of the weight matrices. Biases, normalization and batch normalization values stay 32-bit
        /// </summary>
        public WeightPrecision Precision { get; set; } = WeightPrecision.Single;

        /// <summary>
        /// Weights whose magnitude is below this value are removed. Neurons left without inputs become constants that
        /// are added to the biases of the neurons they feed, and neurons nothing reads any more are removed. Matrices
        /// that end up mostly zero are stored sparsely. 0 keeps every weight
        /// </summary>
        public float PruneThreshold { get; set; }

        /// <summary>
        /// Share of the non-zero weights, between 0 and 1, to remove: the smallest in magnitude across the whole network
        /// (global magnitude pruning). Combined with <see cref="PruneThreshold"/>, the larger of the two thresholds applies
        /// </summary>
        public double PruneFraction { get; set; }

        /// <summary>
        /// The magnitude below which weights are removed, given all weights of a network
        /// </summary>
        internal float Threshold(IEnumerable<float[]> weights)
        {
            if (PruneThreshold < 0 || float.IsNaN(PruneThreshold))
                throw new ArgumentOutOfRangeException(nameof(PruneThreshold), PruneThreshold, "The threshold must be 0 or more");
            if (PruneFraction < 0 || PruneFraction > 1 || double.IsNaN(PruneFraction))
                throw new ArgumentOutOfRangeException(nameof(PruneFraction), PruneFraction, "The fraction must be between 0 and 1");
            if (PruneFraction == 0) return PruneThreshold;

            // zeros are no weights: graph plans pad their matrices with them
            float[] magnitudes = weights.SelectMany(x => x).Where(x => x != 0).Select(Math.Abs).ToArray();
            if (magnitudes.Length == 0) return PruneThreshold;

            Array.Sort(magnitudes);
            int removed = (int)Math.Round(PruneFraction * magnitudes.Length);
            float threshold = removed >= magnitudes.Length ? float.PositiveInfinity : magnitudes[removed];
            if (removed > 0 && threshold == magnitudes[removed - 1]) threshold = MathF.BitIncrement(threshold);
            return Math.Max(PruneThreshold, removed == 0 ? 0 : threshold);
        }
    }
}
