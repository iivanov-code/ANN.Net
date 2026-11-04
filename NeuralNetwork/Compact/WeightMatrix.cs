using System;
using System.IO;
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

        [ThreadStatic]
        private static int[] rowScratch;

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

        public void Write(BinaryWriter writer)
        {
            writer.Write(Rows);
            writer.Write(Columns);
            writer.Write((byte)Precision);
            writer.WriteArray(single);
            writer.WriteArray(half);
            writer.WriteArray(int8);
            writer.WriteArray(scales);
            writer.WriteArray(rowStarts);
            writer.WriteArray(shortColumns);
            writer.WriteArray(columns);
        }

        public static WeightMatrix Read(BinaryReader reader)
        {
            int rows = reader.ReadInt32(), columnCount = reader.ReadInt32();
            var precision = (WeightPrecision)reader.ReadByte();
            CompactIO.Require(rows >= 0 && columnCount >= 0 && Enum.IsDefined(precision), "Invalid weight matrix");

            float[] single = reader.ReadArray<float>(true);
            Half[] half = reader.ReadArray<Half>(true);
            sbyte[] int8 = reader.ReadArray<sbyte>(true);
            float[] scales = reader.ReadArray<float>(true);
            int[] rowStarts = reader.ReadArray<int>(true);
            ushort[] shortColumns = reader.ReadArray<ushort>(true);
            int[] columns = reader.ReadArray<int>(true);

            long stored = precision switch
            {
                WeightPrecision.Single => single?.Length ?? -1,
                WeightPrecision.Half => half?.Length ?? -1,
                _ => int8?.Length ?? -1
            };
            int formats = (single != null ? 1 : 0) + (half != null ? 1 : 0) + (int8 != null ? 1 : 0);
            CompactIO.Require(stored >= 0 && formats == 1, "The weights are missing or stored in more than one format");
            CompactIO.Require((precision == WeightPrecision.Int8) == (scales != null) && (scales == null || scales.Length == columnCount),
                              "Int8 weights need one scale per column");

            if (rowStarts == null)
            {
                CompactIO.Require(stored == (long)rows * columnCount && shortColumns == null && columns == null, "Dense weights do not fill the matrix");
            }
            else
            {
                CompactIO.Require(rowStarts.Length == rows + 1 && rowStarts[0] == 0 && rowStarts[rows] == stored, "Invalid sparse rows");
                for (int r = 0; r < rows; r++) CompactIO.Require(rowStarts[r] <= rowStarts[r + 1], "Invalid sparse rows");

                CompactIO.Require((shortColumns == null) != (columns == null), "Sparse weights need column indexes");
                long indexes = shortColumns?.Length ?? columns.Length;
                CompactIO.Require(indexes == stored, "Sparse weights need one column index per weight");
                for (int k = 0; k < indexes; k++)
                {
                    int column = shortColumns != null ? shortColumns[k] : columns[k];
                    CompactIO.Require(column >= 0 && column < columnCount, "Sparse column index out of range");
                }
            }

            return new WeightMatrix(rows, columnCount, precision, single, half, int8, scales, rowStarts, shortColumns, columns);
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
                {
                    // the rows with a non-zero input, four at a time: one load and store of the outputs for four rows
                    if (rowScratch == null || rowScratch.Length < Rows) rowScratch = new int[Math.Max(Rows, 256)];
                    int[] rows = rowScratch;
                    int used = 0;
                    for (int r = 0; r < Rows; r++)
                    {
                        if (input[r] != 0) rows[used++] = r;
                    }

                    int k = 0;
                    for (; k + 4 <= used; k += 4)
                    {
                        int r0 = rows[k], r1 = rows[k + 1], r2 = rows[k + 2], r3 = rows[k + 3];
                        Axpy4(input[r0], single.AsSpan(r0 * Columns, Columns), input[r1], single.AsSpan(r1 * Columns, Columns),
                              input[r2], single.AsSpan(r2 * Columns, Columns), input[r3], single.AsSpan(r3 * Columns, Columns), output);
                    }

                    for (; k < used; k++) Axpy(input[rows[k]], single.AsSpan(rows[k] * Columns, Columns), output);
                    break;
                }

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

        /// <summary>
        /// <see cref="MultiplyAdd(ReadOnlySpan{float}, Span{float})"/> for <paramref name="count"/> samples, sample b
        /// reading <paramref name="inputs"/> from b * <paramref name="inputStride"/> and writing
        /// <paramref name="outputs"/> from b * <paramref name="outputStride"/>. Every weight row is read (and converted)
        /// once for all samples; the results are bit for bit those of one sample at a time
        /// </summary>
        public void MultiplyAddBatch(ReadOnlySpan<float> inputs, int inputStride, Span<float> outputs, int outputStride, int count)
        {
            // 32-bit rows need no conversion, so there is nothing to share between samples
            if (IsSparse || count == 1 || Precision == WeightPrecision.Single)
            {
                for (int b = 0; b < count; b++)
                {
                    MultiplyAdd(inputs.Slice(b * inputStride, Rows), outputs.Slice(b * outputStride, Columns));
                }

                return;
            }

            Span<float> work = Scratch(Columns * (count + 1));
            Span<float> row = work.Slice(0, Columns);
            Span<float> sums = work.Slice(Columns, Columns * count);
            if (Precision == WeightPrecision.Int8) sums.Clear();

            for (int r = 0; r < Rows; r++)
            {
                bool used = false;
                for (int b = 0; b < count && !used; b++) used = inputs[(b * inputStride) + r] != 0;
                if (!used) continue;

                ReadOnlySpan<float> weights;
                switch (Precision)
                {
                    case WeightPrecision.Single:
                        weights = single.AsSpan(r * Columns, Columns);
                        break;
                    case WeightPrecision.Half:
                        ToSingle(half.AsSpan(r * Columns, Columns), row);
                        weights = row;
                        break;
                    default:
                        ToSingle(int8.AsSpan(r * Columns, Columns), row);
                        weights = row;
                        break;
                }

                for (int b = 0; b < count; b++)
                {
                    float x = inputs[(b * inputStride) + r];
                    if (x == 0) continue;

                    Axpy(x, weights, Precision == WeightPrecision.Int8 ? sums.Slice(b * Columns, Columns) : outputs.Slice(b * outputStride, Columns));
                }
            }

            if (Precision != WeightPrecision.Int8) return;

            for (int b = 0; b < count; b++)
            {
                Span<float> output = outputs.Slice(b * outputStride, Columns);
                ReadOnlySpan<float> sum = sums.Slice(b * Columns, Columns);
                for (int c = 0; c < Columns; c++) output[c] += sum[c] * scales[c];
            }
        }

        /// <summary>
        /// The integers as floats, unscaled
        /// </summary>
        private static void ToSingle(ReadOnlySpan<sbyte> source, Span<float> target)
        {
            int length = source.Length, i = 0;
            if (Vector.IsHardwareAccelerated && length >= Vector<sbyte>.Count)
            {
                int lanes = Vector<float>.Count;
                ref sbyte from = ref MemoryMarshal.GetReference(source);
                ref float to = ref MemoryMarshal.GetReference(target);
                for (; i <= length - Vector<sbyte>.Count; i += Vector<sbyte>.Count)
                {
                    Vector.Widen(Vector.LoadUnsafe(ref from, (nuint)i), out Vector<short> low, out Vector<short> high);
                    Vector.Widen(low, out Vector<int> q0, out Vector<int> q1);
                    Vector.Widen(high, out Vector<int> q2, out Vector<int> q3);
                    Vector.ConvertToSingle(q0).StoreUnsafe(ref to, (nuint)i);
                    Vector.ConvertToSingle(q1).StoreUnsafe(ref to, (nuint)(i + lanes));
                    Vector.ConvertToSingle(q2).StoreUnsafe(ref to, (nuint)(i + (2 * lanes)));
                    Vector.ConvertToSingle(q3).StoreUnsafe(ref to, (nuint)(i + (3 * lanes)));
                }
            }

            for (; i < length; i++) target[i] = source[i];
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

        /// <summary>
        /// Four <see cref="Axpy"/> in one pass, adding in the same order
        /// </summary>
        private static void Axpy4(float x0, ReadOnlySpan<float> row0, float x1, ReadOnlySpan<float> row1,
                                  float x2, ReadOnlySpan<float> row2, float x3, ReadOnlySpan<float> row3, Span<float> output)
        {
            int length = row0.Length, i = 0;
            if (Vector.IsHardwareAccelerated && length >= Vector<float>.Count)
            {
                Vector<float> f0 = new Vector<float>(x0), f1 = new Vector<float>(x1), f2 = new Vector<float>(x2), f3 = new Vector<float>(x3);
                ref float s0 = ref MemoryMarshal.GetReference(row0);
                ref float s1 = ref MemoryMarshal.GetReference(row1);
                ref float s2 = ref MemoryMarshal.GetReference(row2);
                ref float s3 = ref MemoryMarshal.GetReference(row3);
                ref float target = ref MemoryMarshal.GetReference(output);
                for (; i <= length - Vector<float>.Count; i += Vector<float>.Count)
                {
                    nuint at = (nuint)i;
                    Vector<float> sum = Vector.LoadUnsafe(ref target, at) + (f0 * Vector.LoadUnsafe(ref s0, at));
                    sum += f1 * Vector.LoadUnsafe(ref s1, at);
                    sum += f2 * Vector.LoadUnsafe(ref s2, at);
                    sum += f3 * Vector.LoadUnsafe(ref s3, at);
                    sum.StoreUnsafe(ref target, at);
                }
            }

            for (; i < length; i++)
            {
                float sum = output[i] + (x0 * row0[i]);
                sum += x1 * row1[i];
                sum += x2 * row2[i];
                sum += x3 * row3[i];
                output[i] = sum;
            }
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
