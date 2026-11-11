using System;
using ANN.Net.LanguageModels;
using Xunit;

namespace ANN.Net.Tests.LanguageModels
{
    public class TokenSamplerTests
    {
        private static readonly Quad[] DISTRIBUTION = { 0.5f, 0.3f, 0.15f, 0.05f };

        [Fact]
        public void Sample_ZeroTemperature_ReturnsMostLikelyToken()
        {
            for (int i = 0; i < 20; i++)
            {
                Assert.Equal(1, TokenSampler.Sample(new Quad[] { 0.1f, 0.6f, 0.3f }, 0, 0, null));
            }
        }

        [Fact]
        public void Sample_TopKOne_ReturnsMostLikelyToken()
        {
            Random random = new Random(1);

            for (int i = 0; i < 50; i++)
            {
                Assert.Equal(0, TokenSampler.Sample(DISTRIBUTION, 1, 1, random));
            }
        }

        [Fact]
        public void Sample_TopK_NeverLeavesTheTopTokens()
        {
            Random random = new Random(2);

            for (int i = 0; i < 500; i++)
            {
                int token = TokenSampler.Sample(DISTRIBUTION, 1, 2, random);
                Assert.True(token == 0 || token == 1, $"Drew token {token}");
            }
        }

        [Fact]
        public void Sample_UnitTemperature_FollowsTheDistribution()
        {
            Random random = new Random(3);
            int[] draws = new int[4];

            for (int i = 0; i < 10000; i++)
            {
                draws[TokenSampler.Sample(DISTRIBUTION, 1, 0, random)]++;
            }

            Assert.InRange(draws[0], 4700, 5300);
            Assert.InRange(draws[3], 350, 650);
        }

        [Fact]
        public void Sample_LowTemperature_SharpensTheDistribution()
        {
            Random random = new Random(4);
            int[] draws = new int[4];

            for (int i = 0; i < 10000; i++)
            {
                draws[TokenSampler.Sample(DISTRIBUTION, 0.2f, 0, random)]++;
            }

            Assert.True(draws[0] > 9000, $"Most likely token drawn {draws[0]} times");
        }

        [Fact]
        public void Sample_SameSeed_IsReproducible()
        {
            int[] first = new int[20];
            int[] second = new int[20];
            Random firstRandom = new Random(5);
            Random secondRandom = new Random(5);

            for (int i = 0; i < 20; i++)
            {
                first[i] = TokenSampler.Sample(DISTRIBUTION, 1, 0, firstRandom);
                second[i] = TokenSampler.Sample(DISTRIBUTION, 1, 0, secondRandom);
            }

            Assert.Equal(first, second);
        }

        [Fact]
        public void Sample_InvalidArguments_Throw()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => TokenSampler.Sample(DISTRIBUTION, -1, 0, new Random(6)));
            Assert.Throws<ArgumentNullException>(() => TokenSampler.Sample(DISTRIBUTION, 1, 0, null));
            Assert.Throws<ArgumentException>(() => TokenSampler.Sample(Array.Empty<Quad>(), 0, 0, null));
        }
    }
}
