using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ANN.Net.Abstractions.Enums;

namespace ANN.Net.Compact
{
    /// <summary>
    /// Rows x columns weights of a compact network (one row per input, one column per neuron or filter) in 32-bit,
    /// 16-bit or 8-bit precision, stored densely or, when most weights are zero and that is smaller, as the non-zero
    /// weights of every row with their column indexes. Int8 weights have one scale per column
    /// </summary>
    internal sealed class WeightMatrix
    {
        /// <summary>
        /// Sparse storage is used when it needs at most this share of the dense storage; dense rows run with SIMD
        /// and are faster per stored weight
        /// </summary>
        private const double SparseShare = 0.5;

        [ThreadStatic]
        private static float[] scratch;

        private readonly float[] single;
        private readonly Half[] half;
        private readonly sbyte[] int8;
        private readonly float[] scales;
        private readonly int[] rowStarts;
        private readonly ushort[] shortColumns;
        private readonly int[] columns;

        private WeightMatrix(int rows, int columnCount, WeightPrecision precision, float[] single, Half[] half, sbyte[] int8, float[] scales,
                             int[] rowStarts, ushort[] shortColumns, int[] columns)
        {
            Rows = rows;
            Columns = columnCount;
            Precision = precision;
            this.single = single;
            this.half = half;
            this.int8 = int8;
            this.scales = scales;
            this.rowStarts = rowStarts;
            this.shortColumns = shortColumns;
            this.columns = columns;
        }

        public int Rows { get; }

        public int Columns { get; }

        public WeightPrecision Precision { get; }

        public bool IsSparse => rowStarts != null;

        /// <summary>
        /// Number of stored weights: all of them when dense, the non-zero ones when sparse
        /// </summary>
        public long StoredValues => single?.Length ?? half?.Length ?? int8.Length;

        public long SizeInBytes => (StoredValues * BytesPerValue(Precision))
                                   + ((scales?.Length ?? 0) * 4L)
                                   + ((rowStarts?.Length ?? 0) * 4L)
                                   + ((shortColumns?.Length ?? 0) * 2L)
                                   + ((columns?.Length ?? 0) * 4L);

        internal float[] Scales => scales;

        internal int[] RowStarts => rowStarts;

        private static int BytesPerValue(WeightPrecision precision) => precision switch
        {
            WeightPrecision.Single => 4,
            WeightPrecision.Half => 2,
            _ => 1
        };

        /// <param name="weights">Row by row: the weight from input r to column c is at r * columns + c</param>
        public static WeightMatrix Create(float[] weights, int rows, int columnCount, WeightPrecision precision)
        {
            ArgumentNullException.ThrowIfNull(weights);
            if (weights.Length != (long)rows * columnCount)
                throw new ArgumentException($"Expected {(long)rows * columnCount} weights but found {weights.Length}", nameof(weights));

            float[] scales = precision == WeightPrecision.Int8 ? Int8Scales(weights, columnCount, 1) : null;

            long nonZero = 0;
            foreach (float weight in weights)
            {
                if (weight != 0) nonZero++;
            }

            int indexBytes = columnCount <= ushort.MaxValue + 1 ? 2 : 4;
            long dense = weights.LongLength * BytesPerValue(precision);
            long sparse = (nonZero * (BytesPerValue(precision) + indexBytes)) + ((rows + 1L) * 4);
            if (sparse > dense * SparseShare)
            {
                return new WeightMatrix(rows, columnCount, precision,
                                        precision == WeightPrecision.Single ? (float[])weights.Clone() : null,
                                        precision == WeightPrecision.Half ? ToHalf(weights) : null,
                                        precision == WeightPrecision.Int8 ? ToInt8(weights, scales, columnCount, 1) : null,
                                        scales, null, null, null);
            }

            var starts = new int[rows + 1];
            var kept = new float[nonZero];
            var columnIndexes = new int[nonZero];
            int k = 0;
            for (int r = 0; r < rows; r++)
            {
                starts[r] = k;
                for (int c = 0; c < columnCount; c++)
                {
                    float weight = weights[((long)r * columnCount) + c];
                    if (weight == 0) continue;

                    kept[k] = weight;
                    columnIndexes[k++] = c;
                }
            }

            starts[rows] = k;
            float[] keptScales = null;
            sbyte[] keptInt8 = null;
            if (precision == WeightPrecision.Int8)
            {
                keptScales = scales;
                keptInt8 = new sbyte[nonZero];
                for (int i = 0; i < kept.Length; i++) keptInt8[i] = Quantize(kept[i], scales[columnIndexes[i]]);
            }

            return new WeightMatrix(rows, columnCount, precision,
                                    precision == WeightPrecision.Single ? kept : null,
                                    precision == WeightPrecision.Half ? ToHalf(kept) : null,
                                    keptInt8, keptScales, starts,
                                    indexBytes == 2 ? Array.ConvertAll(columnIndexes, x => (ushort)x) : null,
                                    indexBytes == 4 ? columnIndexes : null);
        }

        /// <summary>
        /// Symmetric int8 scale of every channel: max|w| / 127, so that the largest weight maps to ±127. The channel of
        /// element k is (k / <paramref name="stride"/>) % <paramref name="channels"/>
        /// </summary>
        public static float[] Int8Scales(float[] weights, int channels, int stride)
        {
            var max = new float[channels];
            for (int k = 0; k < weights.Length; k++)
            {
                int channel = (k / stride) % channels;
                max[channel] = Math.Max(max[channel], Math.Abs(weights[k]));
            }

            return Array.ConvertAll(max, m => m == 0 ? 1f : m / 127f);
        }

        public static sbyte Quantize(float weight, float scale) => (sbyte)Math.Clamp(MathF.Round(weight / scale), -127, 127);

        public static sbyte[] ToInt8(float[] weights, float[] scales, int channels, int stride)
        {
            var result = new sbyte[weights.Length];
            for (int k = 0; k < weights.Length; k++) result[k] = Quantize(weights[k], scales[(k / stride) % channels]);
            return result;
        }

        public static Half[] ToHalf(float[] weights) => Array.ConvertAll(weights, x => (Half)x);

        /// <summary>
        /// The stored weight from input <paramref name="row"/> to <paramref name="column"/>, as the matrix computes with it
        /// </summary>
        public float this[int row, int column]
        {
            get
            {
                if (!IsSparse) return Value((row * Columns) + column, column);

                for (int k = rowStarts[row]; k < rowStarts[row + 1]; k++)
                {
                    if (ColumnAt(k) == column) return Value(k, column);
                }

                return 0;
            }
        }

        private int ColumnAt(int k) => shortColumns != null ? shortColumns[k] : columns[k];

        private float Value(int k, int column) => Precision switch
        {
            WeightPrecision.Single => single[k],
            WeightPrecision.Half => (float)half[k],
            _ => int8[k] * scales[column]
        };

        private static Span<float> Scratch(int length)
        {
            if (scratch == null || scratch.Length < length) scratch = new float[Math.Max(length, 256)];
            return scratch.AsSpan(0, length);
        }

        /// <summary>
        /// output[c] += Σ_r input[r] * W[r, c]; rows whose input is 0 are skipped
        /// </summary>
        public void MultiplyAdd(ReadOnlySpan<float> input, Span<float> output)
        {
            input = input.Slice(0, Rows);
            output = output.Slice(0, Columns);

            if (IsSparse)
            {
                SparseMultiplyAdd(input, output);
                return;
            }

            switch (Precision)
            {
                case WeightPrecision.Single:
                    for (int r = 0; r < Rows; r++)
                    {
                        float x = input[r];
                        if (x != 0) Axpy(x, single.AsSpan(r * Columns, Columns), output);
                    }

                    break;

                case WeightPrecision.Half:
                {
                    Span<float> row = Scratch(Columns);
                    for (int r = 0; r < Rows; r++)
                    {
                        float x = input[r];
                        if (x == 0) continue;

                        ToSingle(half.AsSpan(r * Columns, Columns), row);
                        Axpy(x, row, output);
                    }

                    break;
                }

                default:
                {
                    // integer weights are summed first and scaled once per column
                    Span<float> sums = Scratch(Columns);
                    sums.Clear();
                    for (int r = 0; r < Rows; r++)
                    {
                        float x = input[r];
                        if (x != 0) AxpyInt8(x, int8.AsSpan(r * Columns, Columns), sums);
                    }

                    for (int c = 0; c < Columns; c++) output[c] += sums[c] * scales[c];
                    break;
                }
            }
        }

        private void SparseMultiplyAdd(ReadOnlySpan<float> input, Span<float> output)
        {
            for (int r = 0; r < Rows; r++)
            {
                float x = input[r];
                if (x == 0) continue;

                int end = rowStarts[r + 1];
                for (int k = rowStarts[r]; k < end; k++)
                {
                    int c = ColumnAt(k);
                    output[c] += x * Value(k, c);
                }
            }
        }

        private static void Axpy(float x, ReadOnlySpan<float> row, Span<float> output)
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

        private static void AxpyInt8(float x, ReadOnlySpan<sbyte> row, Span<float> sums)
        {
            int length = row.Length, i = 0;
            if (Vector.IsHardwareAccelerated && length >= Vector<sbyte>.Count)
            {
                var factor = new Vector<float>(x);
                int lanes = Vector<float>.Count;
                ref sbyte source = ref MemoryMarshal.GetReference(row);
                ref float target = ref MemoryMarshal.GetReference(sums);
                for (; i <= length - Vector<sbyte>.Count; i += Vector<sbyte>.Count)
                {
                    Vector.Widen(Vector.LoadUnsafe(ref source, (nuint)i), out Vector<short> low, out Vector<short> high);
                    Vector.Widen(low, out Vector<int> q0, out Vector<int> q1);
                    Vector.Widen(high, out Vector<int> q2, out Vector<int> q3);

                    nuint at = (nuint)i;
                    (Vector.LoadUnsafe(ref target, at) + (factor * Vector.ConvertToSingle(q0))).StoreUnsafe(ref target, at);
                    at += (nuint)lanes;
                    (Vector.LoadUnsafe(ref target, at) + (factor * Vector.ConvertToSingle(q1))).StoreUnsafe(ref target, at);
                    at += (nuint)lanes;
                    (Vector.LoadUnsafe(ref target, at) + (factor * Vector.ConvertToSingle(q2))).StoreUnsafe(ref target, at);
                    at += (nuint)lanes;
                    (Vector.LoadUnsafe(ref target, at) + (factor * Vector.ConvertToSingle(q3))).StoreUnsafe(ref target, at);
                }
            }

            for (; i < length; i++) sums[i] += x * row[i];
        }

        /// <summary>
        /// IEEE half to single conversion, several values at a time: normal numbers move their exponent and mantissa,
        /// infinities and NaNs keep an all-ones exponent, subnormal halves are normal singles computed by multiplication
        /// </summary>
        internal static void ToSingle(ReadOnlySpan<Half> source, Span<float> target)
        {
            int length = source.Length, i = 0;
            if (Vector.IsHardwareAccelerated && length >= Vector<ushort>.Count)
            {
                ReadOnlySpan<ushort> bits = MemoryMarshal.Cast<Half, ushort>(source);
                ref ushort from = ref MemoryMarshal.GetReference(bits);
                ref uint to = ref Unsafe.As<float, uint>(ref MemoryMarshal.GetReference(target));
                for (; i <= length - Vector<ushort>.Count; i += Vector<ushort>.Count)
                {
                    Vector.Widen(Vector.LoadUnsafe(ref from, (nuint)i), out Vector<uint> low, out Vector<uint> high);
                    Convert(low).StoreUnsafe(ref to, (nuint)i);
                    Convert(high).StoreUnsafe(ref to, (nuint)(i + Vector<uint>.Count));
                }
            }

            for (; i < length; i++) target[i] = (float)source[i];
        }

        private static Vector<uint> Convert(Vector<uint> half)
        {
            Vector<uint> sign = (half & new Vector<uint>(0x8000)) << 16;
            Vector<uint> magnitude = half & new Vector<uint>(0x7FFF);
            Vector<uint> normal = (magnitude << 13) + new Vector<uint>(0x38000000);
            Vector<uint> special = normal + new Vector<uint>(0x38000000);
            Vector<uint> subnormal = Vector.AsVectorUInt32(Vector.ConvertToSingle(Vector.AsVectorInt32(magnitude)) * new Vector<float>(5.9604645e-8f));

            Vector<uint> result = Vector.ConditionalSelect(Vector.GreaterThanOrEqual(magnitude, new Vector<uint>(0x7C00)), special, normal);
            result = Vector.ConditionalSelect(Vector.LessThan(magnitude, new Vector<uint>(0x0400)), subnormal, result);
            return result | sign;
        }
    }
}
