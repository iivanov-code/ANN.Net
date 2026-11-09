using System;
using System.Collections.Generic;
using System.Linq;

namespace ANN.Net.Abstractions.Spiking
{
    /// <summary>
    /// Normalization of the inputs of a spiking network: every input becomes (input - offset) * scale. An input
    /// neuron fires at most once per time step, so the result should lie in [0, 1], or in [-1, 1] for an input
    /// activation that fires negative spikes; a larger value fires every step and is cut off
    /// </summary>
    public sealed class InputScaling
    {
        private readonly float[] offset;
        private readonly float[] scale;

        public InputScaling(IReadOnlyList<float> offset, IReadOnlyList<float> scale)
        {
            ArgumentNullException.ThrowIfNull(offset);
            ArgumentNullException.ThrowIfNull(scale);
            if (offset.Count != scale.Count) throw new ArgumentException("Offset and scale must have the same length", nameof(scale));
            if (offset.Any(x => !float.IsFinite(x)) || scale.Any(x => !float.IsFinite(x)))
                throw new ArgumentException("Offset and scale must be finite numbers", nameof(scale));

            this.offset = offset.ToArray();
            this.scale = scale.ToArray();
        }

        public int Count => offset.Length;

        public IReadOnlyList<float> Offset => offset;

        public IReadOnlyList<float> Scale => scale;

        /// <summary>
        /// Maps every input from its smallest to its largest value in the samples onto [0, 1]. An input with a single
        /// value becomes 0
        /// </summary>
        public static InputScaling MinMax(IEnumerable<float[]> samples)
        {
            var (minimum, maximum) = Bounds(samples);
            return new InputScaling(minimum, minimum.Select((m, i) => maximum[i] > m ? 1 / (maximum[i] - m) : 0).ToArray());
        }

        /// <summary>
        /// Divides every input by its largest magnitude in the samples, giving [-1, 1] with 0 staying 0; for input
        /// activations that fire negative spikes
        /// </summary>
        public static InputScaling MaxAbs(IEnumerable<float[]> samples)
        {
            var (minimum, maximum) = Bounds(samples);
            return new InputScaling(new float[minimum.Length], minimum.Select((m, i) =>
            {
                float magnitude = Math.Max(Math.Abs(m), Math.Abs(maximum[i]));
                return magnitude > 0 ? 1 / magnitude : 0;
            }).ToArray());
        }

        /// <summary>
        /// Maps [minimum, maximum] onto [0, 1] for every one of <paramref name="count"/> inputs
        /// </summary>
        public static InputScaling Range(float minimum, float maximum, int count)
        {
            if (!(maximum > minimum)) throw new ArgumentOutOfRangeException(nameof(maximum), "The maximum must be greater than the minimum");
            ArgumentOutOfRangeException.ThrowIfNegative(count);

            return new InputScaling(Enumerable.Repeat(minimum, count).ToArray(), Enumerable.Repeat(1 / (maximum - minimum), count).ToArray());
        }

        public void Apply(ReadOnlySpan<float> inputs, Span<float> scaled)
        {
            if (inputs.Length != offset.Length) throw new ArgumentException($"Expected {offset.Length} inputs, got {inputs.Length}", nameof(inputs));

            for (int i = 0; i < offset.Length; i++)
            {
                scaled[i] = (inputs[i] - offset[i]) * scale[i];
            }
        }

        private static (float[] Minimum, float[] Maximum) Bounds(IEnumerable<float[]> samples)
        {
            ArgumentNullException.ThrowIfNull(samples);

            float[] minimum = null, maximum = null;
            foreach (float[] sample in samples)
            {
                if (minimum == null)
                {
                    minimum = (float[])sample.Clone();
                    maximum = (float[])sample.Clone();
                    continue;
                }

                if (sample.Length != minimum.Length) throw new ArgumentException("All samples must have the same length", nameof(samples));

                for (int i = 0; i < sample.Length; i++)
                {
                    minimum[i] = Math.Min(minimum[i], sample[i]);
                    maximum[i] = Math.Max(maximum[i], sample[i]);
                }
            }

            if (minimum == null) throw new ArgumentException("At least one sample is needed", nameof(samples));
            return (minimum, maximum);
        }
    }
}
