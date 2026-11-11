using System;
using System.Numerics;
using System.Runtime.InteropServices;

namespace ANN.Net.Recurrent
{
    internal static class RecurrentMath
    {
        /// <summary>
        /// output += x * row
        /// </summary>
        public static void Axpy(float x, ReadOnlySpan<float> row, Span<float> output)
        {
            int length = row.Length, i = 0;
            if (Vector.IsHardwareAccelerated && length >= Vector<float>.Count)
            {
                var factor = new Vector<float>(x);
                ref float source = ref MemoryMarshal.GetReference(row);
                ref float target = ref MemoryMarshal.GetReference(output);
                for (; i <= length - Vector<float>.Count; i += Vector<float>.Count)
                {
                    (Vector.LoadUnsafe(ref target, (nuint)i) + (factor * Vector.LoadUnsafe(ref source, (nuint)i))).StoreUnsafe(ref target, (nuint)i);
                }
            }

            for (; i < length; i++) output[i] += x * row[i];
        }

        public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            int length = a.Length, i = 0;
            float sum = 0;
            if (Vector.IsHardwareAccelerated && length >= Vector<float>.Count)
            {
                Vector<float> sums = Vector<float>.Zero;
                ref float left = ref MemoryMarshal.GetReference(a);
                ref float right = ref MemoryMarshal.GetReference(b);
                for (; i <= length - Vector<float>.Count; i += Vector<float>.Count)
                {
                    sums += Vector.LoadUnsafe(ref left, (nuint)i) * Vector.LoadUnsafe(ref right, (nuint)i);
                }

                sum = Vector.Sum(sums);
            }

            for (; i < length; i++) sum += a[i] * b[i];
            return sum;
        }

        public static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

        /// <summary>
        /// sums += inputs * weights[rows, offset .. offset + sums.Length], the weights stored row by row with
        /// <paramref name="columns"/> columns and one row per input, starting at <paramref name="firstRow"/>
        /// </summary>
        public static void Affine(ReadOnlySpan<float> inputs, ReadOnlySpan<float> weights, int firstRow, int columns, int offset, Span<float> sums)
        {
            for (int k = 0; k < inputs.Length; k++)
            {
                Axpy(inputs[k], weights.Slice(((firstRow + k) * columns) + offset, sums.Length), sums);
            }
        }

        /// <summary>
        /// gradients[rows, offset ..] += inputs (outer) deltas, and inputGradients += weights[rows, offset ..] * deltas
        /// </summary>
        public static void AffineBackward(ReadOnlySpan<float> inputs, ReadOnlySpan<float> weights, Span<float> gradients, int firstRow, int columns, int offset,
                                          ReadOnlySpan<float> deltas, Span<float> inputGradients)
        {
            for (int k = 0; k < inputs.Length; k++)
            {
                int start = ((firstRow + k) * columns) + offset;
                Axpy(inputs[k], deltas, gradients.Slice(start, deltas.Length));
                inputGradients[k] += Dot(weights.Slice(start, deltas.Length), deltas);
            }
        }
    }
}
