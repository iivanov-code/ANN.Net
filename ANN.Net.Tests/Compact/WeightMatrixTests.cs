using System;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Compact;
using Xunit;

namespace ANN.Net.Tests.Compact
{
    public class WeightMatrixTests
    {
        private static float[] Weights(int rows, int columns, double zeros, int seed)
        {
            var random = new Random(seed);
            return Enumerable.Range(0, rows * columns).Select(_ => random.NextDouble() < zeros ? 0f : (float)((random.NextDouble() * 2) - 1)).ToArray();
        }

        [Theory]
        [InlineData(WeightPrecision.Single, 0.0)]
        [InlineData(WeightPrecision.Half, 0.0)]
        [InlineData(WeightPrecision.Int8, 0.0)]
        [InlineData(WeightPrecision.Single, 0.9)]
        [InlineData(WeightPrecision.Half, 0.9)]
        [InlineData(WeightPrecision.Int8, 0.9)]
        public void MultiplyAdd_ComputesWithTheStoredWeights(WeightPrecision precision, double zeros)
        {
            // 37 columns: SIMD lanes and a remainder
            const int rows = 23, columns = 37;
            float[] weights = Weights(rows, columns, zeros, 1);
            var matrix = WeightMatrix.Create(weights, rows, columns, precision);
            Assert.Equal(zeros > 0.5, matrix.IsSparse);

            var random = new Random(2);
            float[] input = Enumerable.Range(0, rows).Select(i => i % 5 == 0 ? 0f : (float)((random.NextDouble() * 4) - 2)).ToArray();
            float[] output = Enumerable.Range(0, columns).Select(i => i * 0.5f).ToArray();
            float[] expected = (float[])output.Clone();
            for (int c = 0; c < columns; c++)
            {
                for (int r = 0; r < rows; r++) expected[c] += input[r] * matrix[r, c];
            }

            matrix.MultiplyAdd(input, output);

            for (int c = 0; c < columns; c++) Assert.Equal(expected[c], output[c], 1e-4f);
        }

        [Theory]
        [InlineData(WeightPrecision.Single, 0)]
        [InlineData(WeightPrecision.Half, 1e-3)]
        [InlineData(WeightPrecision.Int8, 0.5 / 127)]
        public void StoredWeights_AreCloseToTheOriginals(WeightPrecision precision, double relativeError)
        {
            const int rows = 40, columns = 9;
            float[] weights = Weights(rows, columns, 0, 3);
            var matrix = WeightMatrix.Create(weights, rows, columns, precision);

            for (int c = 0; c < columns; c++)
            {
                float max = Enumerable.Range(0, rows).Max(r => Math.Abs(weights[(r * columns) + c]));
                for (int r = 0; r < rows; r++)
                {
                    float original = weights[(r * columns) + c];
                    // half: relative to the weight; int8: relative to the largest weight of the column
                    double bound = precision == WeightPrecision.Int8 ? relativeError * max * 1.0001 : (relativeError * Math.Abs(original)) + 1e-7;
                    Assert.True(Math.Abs(matrix[r, c] - original) <= bound, $"{original} stored as {matrix[r, c]}");
                }
            }
        }

        [Fact]
        public void HalfConversion_MatchesTheFrameworkForEveryHalf()
        {
            Half[] all = Enumerable.Range(0, 65536).Select(i => BitConverter.UInt16BitsToHalf((ushort)i)).ToArray();
            var converted = new float[all.Length];
            WeightMatrix.ToSingle(all, converted);

            for (int i = 0; i < all.Length; i++)
            {
                float expected = (float)all[i];
                Assert.True(float.IsNaN(expected) ? float.IsNaN(converted[i]) : BitConverter.SingleToInt32Bits(expected) == BitConverter.SingleToInt32Bits(converted[i]),
                            $"half 0x{i:X4}: {expected} but converted {converted[i]}");
            }
        }

        [Fact]
        public void Size_ShrinksWithPrecisionAndSparseStorage()
        {
            const int rows = 64, columns = 64;
            float[] dense = Weights(rows, columns, 0, 4);
            Assert.Equal(rows * columns * 4, WeightMatrix.Create(dense, rows, columns, WeightPrecision.Single).SizeInBytes);
            Assert.Equal(rows * columns * 2, WeightMatrix.Create(dense, rows, columns, WeightPrecision.Half).SizeInBytes);
            Assert.Equal((rows * columns) + (columns * 4), WeightMatrix.Create(dense, rows, columns, WeightPrecision.Int8).SizeInBytes);

            float[] sparse = Weights(rows, columns, 0.95, 5);
            int nonZero = sparse.Count(x => x != 0);
            var matrix = WeightMatrix.Create(sparse, rows, columns, WeightPrecision.Single);
            Assert.True(matrix.IsSparse);
            Assert.Equal(nonZero, matrix.StoredValues);
            Assert.Equal((nonZero * (4 + 2)) + ((rows + 1) * 4), matrix.SizeInBytes);
        }
    }
}
