using System;
using System.Collections.Generic;
using ANN.Net.LanguageModels;
using Xunit;

namespace ANN.Net.Tests.LanguageModels
{
    public class TokenWindowsTests
    {
        [Fact]
        public void Build_WrapsInSpecialTokensAndOverlapsByOne()
        {
            List<int[]> windows = TokenWindows.Build(new[] { new[] { 2, 3, 4, 5, 2, 3 } }, 3, 3, 0, 1);

            Assert.Equal(3, windows.Count);
            Assert.Equal(new[] { 0, 2, 3, 4 }, windows[0]);
            Assert.Equal(new[] { 4, 5, 2, 3 }, windows[1]);
            Assert.Equal(new[] { 3, 1 }, windows[2]);
        }

        [Fact]
        public void Build_WithoutSpecialTokens_UsesTheSequenceAsIs()
        {
            List<int[]> windows = TokenWindows.Build(new[] { new[] { 2, 3, 4, 5 } }, 3, 3, null, null);

            Assert.Single(windows);
            Assert.Equal(new[] { 2, 3, 4, 5 }, windows[0]);
        }

        [Fact]
        public void Build_WithSmallerStride_OverlapsMore()
        {
            List<int[]> windows = TokenWindows.Build(new[] { new[] { 2, 3, 4, 5, 2 } }, 3, 1, null, null);

            // once a window reaches the end of the sequence no further window starts
            Assert.Equal(2, windows.Count);
            Assert.Equal(new[] { 2, 3, 4, 5 }, windows[0]);
            Assert.Equal(new[] { 3, 4, 5, 2 }, windows[1]);
        }

        [Fact]
        public void Build_DropsSequencesShorterThanTwoTokens()
        {
            List<int[]> windows = TokenWindows.Build(new[] { new[] { 5 }, Array.Empty<int>() }, 3, 3, null, null);

            Assert.Empty(windows);
        }

        [Fact]
        public void Build_SingleTokenWithEndToken_IsOneWindow()
        {
            List<int[]> windows = TokenWindows.Build(new[] { new[] { 5 } }, 3, 3, null, 1);

            Assert.Single(windows);
            Assert.Equal(new[] { 5, 1 }, windows[0]);
        }

        [Fact]
        public void Build_NullSequence_Throws()
        {
            Assert.Throws<ArgumentException>(() => TokenWindows.Build(new int[][] { null }, 3, 3, null, null));
            Assert.Throws<ArgumentNullException>(() => TokenWindows.Build(null, 3, 3, null, null));
        }
    }
}
