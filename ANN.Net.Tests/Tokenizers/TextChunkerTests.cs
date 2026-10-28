using System.Collections.Generic;
using ANN.Net.Tokenizers;
using Xunit;

namespace ANN.Net.Tests.Tokenizers
{
    public class TextChunkerTests
    {
        [Fact]
        public void Split_AttachesLeadingWhitespaceToTheFollowingWord()
        {
            IList<string> chunks = TextChunker.Split("Hello big  world");

            Assert.Equal(new[] { "Hello", " big", "  world" }, chunks);
        }

        [Fact]
        public void Split_KeepsTrailingWhitespaceAsItsOwnChunk()
        {
            IList<string> chunks = TextChunker.Split("end\n\n");

            Assert.Equal(new[] { "end", "\n\n" }, chunks);
        }

        [Fact]
        public void Split_OfEmptyTextIsEmpty()
        {
            Assert.Empty(TextChunker.Split(string.Empty));
        }

        [Fact]
        public void Split_ConcatenatesBackToTheOriginalText()
        {
            string text = "  Здравей, свят!\tHow are you? 😀\r\n";

            Assert.Equal(text, string.Concat(TextChunker.Split(text)));
        }
    }
}
