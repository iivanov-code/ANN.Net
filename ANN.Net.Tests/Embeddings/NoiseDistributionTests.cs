using System;
using ANN.Net.Embeddings;
using Xunit;

namespace ANN.Net.Tests.Embeddings
{
    public class NoiseDistributionTests
    {
        [Fact]
        public void Sample_NeverDrawsTokensWithZeroCount()
        {
            NoiseDistribution noise = new NoiseDistribution(new long[] { 0, 10, 0, 5, 0 }, 0.75);
            Random random = new Random(1);

            for (int i = 0; i < 2000; i++)
            {
                int token = noise.Sample(random);
                Assert.True(token == 1 || token == 3, $"Drew token {token}");
            }
        }

        [Fact]
        public void Sample_FavoursFrequentTokens()
        {
            NoiseDistribution noise = new NoiseDistribution(new long[] { 1, 100, 10 }, 1);
            Random random = new Random(2);

            int[] draws = new int[3];
            for (int i = 0; i < 10000; i++)
            {
                draws[noise.Sample(random)]++;
            }

            Assert.True(draws[1] > draws[2]);
            Assert.True(draws[2] > draws[0]);
            Assert.InRange(draws[1], 8500, 9500);
        }

        [Fact]
        public void Sample_WithZeroPower_IsUniformOverPresentTokens()
        {
            NoiseDistribution noise = new NoiseDistribution(new long[] { 1, 1000, 0 }, 0);
            Random random = new Random(3);

            int[] draws = new int[3];
            for (int i = 0; i < 10000; i++)
            {
                draws[noise.Sample(random)]++;
            }

            Assert.Equal(0, draws[2]);
            Assert.InRange(draws[0], 4500, 5500);
        }

        [Fact]
        public void Sample_WithoutCounts_Throws()
        {
            NoiseDistribution noise = new NoiseDistribution(new long[] { 0, 0 }, 0.75);

            Assert.False(noise.HasSamples);
            Assert.Throws<InvalidOperationException>(() => noise.Sample(new Random(4)));
        }
    }
}
