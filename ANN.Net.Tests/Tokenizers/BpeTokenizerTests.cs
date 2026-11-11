using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Tokenizers;
using Xunit;

namespace ANN.Net.Tests.Tokenizers
{
    public class BpeTokenizerTests
    {
        private const string CORPUS_LINE = "the cat sat on the mat and the dog sat on the log";

        private static ITokenizer CreateTokenizer(ushort vocabularySize = 300, params string[] specialTokens)
        {
            return NetworkFactory.BuildBpeTokenizer(new BpeTokenizerSettings
            {
                VocabularySize = vocabularySize,
                SpecialTokens = specialTokens.ToList()
            });
        }

        private static IEnumerable<string> Corpus(int repetitions = 20)
        {
            return Enumerable.Repeat(CORPUS_LINE, repetitions);
        }

        [Fact]
        public void Untrained_EncodesToUtf8Bytes()
        {
            ITokenizer tokenizer = CreateTokenizer();

            int[] tokens = tokenizer.Encode("hi ж");

            Assert.Equal(256, tokenizer.VocabularySize);
            Assert.Equal(Encoding.UTF8.GetBytes("hi ж").Select(b => (int)b), tokens);
        }

        [Theory]
        [InlineData("the cat sat")]
        [InlineData("Здравей, свят! Котката седна на постелката.")]
        [InlineData("tabs\tand\r\nnewlines  and emoji 😀🐈")]
        [InlineData("")]
        public void EncodeDecode_RoundTripsBeforeAndAfterTraining(string text)
        {
            ITokenizer tokenizer = CreateTokenizer();
            Assert.Equal(text, tokenizer.Decode(tokenizer.Encode(text)));

            tokenizer.Train(Corpus());
            Assert.Equal(text, tokenizer.Decode(tokenizer.Encode(text)));
        }

        [Fact]
        public void Train_MergesTheMostFrequentPairFirst()
        {
            BpeTokenizer tokenizer = (BpeTokenizer)CreateTokenizer(257);

            // "th" and "at" both appear 4 times per line, more than any other pair;
            // the tie breaks on the lowest pair so ('a','t') is the first merge
            tokenizer.Train(Corpus());

            Assert.Single(tokenizer.Merges);
            Assert.Equal(((int)'a', (int)'t'), tokenizer.Merges[0]);
            Assert.Equal("at", tokenizer.Decode(new[] { 256 }));
        }

        [Fact]
        public void Train_TurnsAFrequentWordIntoASingleToken()
        {
            ITokenizer tokenizer = CreateTokenizer();

            tokenizer.Train(Corpus());

            Assert.Single(tokenizer.Encode(" the"));
            Assert.True(tokenizer.Encode(CORPUS_LINE).Length < Encoding.UTF8.GetByteCount(CORPUS_LINE) / 2);
        }

        [Fact]
        public void Train_NeverMergesAcrossChunkBoundaries()
        {
            ITokenizer tokenizer = CreateTokenizer(400);

            tokenizer.Train(Enumerable.Repeat("x y", 50));

            // "x" and " y" are separate chunks so "x y" can never become one token
            int[] tokens = tokenizer.Encode("x y");
            Assert.Equal(2, tokens.Length);
            Assert.Equal("x", tokenizer.Decode(new[] { tokens[0] }));
            Assert.Equal(" y", tokenizer.Decode(new[] { tokens[1] }));
        }

        [Fact]
        public void Train_StopsWhenNoPairReachesTheMinimumFrequency()
        {
            ITokenizer tokenizer = CreateTokenizer(1000);

            tokenizer.Train(new[] { "abcdefg" });

            Assert.Equal(256, tokenizer.VocabularySize);
        }

        [Fact]
        public void Train_ReachesTheTargetVocabularySizeWhenTheCorpusAllows()
        {
            ITokenizer tokenizer = CreateTokenizer(270);

            tokenizer.Train(Corpus());

            Assert.Equal(270, tokenizer.VocabularySize);
        }

        [Fact]
        public void Train_IsDeterministic()
        {
            ITokenizer first = CreateTokenizer();
            ITokenizer second = CreateTokenizer();

            first.Train(Corpus());
            second.Train(Corpus());

            Assert.Equal(first.VocabularySize, second.VocabularySize);
            Assert.Equal(first.Encode(CORPUS_LINE), second.Encode(CORPUS_LINE));
        }

        [Fact]
        public void Train_CanContinueOnMoreText()
        {
            ITokenizer tokenizer = CreateTokenizer();
            tokenizer.Train(Corpus());
            int sizeAfterFirstCorpus = tokenizer.VocabularySize;

            tokenizer.Train(Enumerable.Repeat("zebra zebra zebra", 10));

            Assert.True(tokenizer.VocabularySize > sizeAfterFirstCorpus);
            Assert.Single(tokenizer.Encode(" zebra"));
            Assert.Single(tokenizer.Encode(" the"));
        }

        [Fact]
        public void SpecialTokens_GetTheIdsAfterTheByteTokens()
        {
            ITokenizer tokenizer = CreateTokenizer(300, "<pad>", "<eos>");

            Assert.Equal(258, tokenizer.VocabularySize);
            Assert.Equal(256, tokenizer.SpecialTokens["<pad>"]);
            Assert.Equal(257, tokenizer.SpecialTokens["<eos>"]);
            Assert.Equal("<eos>", tokenizer.Decode(new[] { 257 }));
        }

        [Fact]
        public void SpecialTokens_AreNeverProducedFromPlainText()
        {
            ITokenizer tokenizer = CreateTokenizer(300, "<eos>");
            tokenizer.Train(Enumerable.Repeat("<eos> <eos> <eos>", 20));

            int[] tokens = tokenizer.Encode("<eos>");

            Assert.DoesNotContain(tokenizer.SpecialTokens["<eos>"], tokens);
            Assert.Equal("<eos>", tokenizer.Decode(tokens));
        }

        [Fact]
        public void SpecialTokens_KeepTheirIdsAfterTraining()
        {
            ITokenizer tokenizer = CreateTokenizer(300, "<pad>");

            tokenizer.Train(Corpus());

            Assert.Equal(256, tokenizer.SpecialTokens["<pad>"]);
            Assert.Equal("<pad>", tokenizer.Decode(new[] { 256 }));
        }

        [Fact]
        public void Decode_RejectsIdsOutsideTheVocabulary()
        {
            ITokenizer tokenizer = CreateTokenizer();

            Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.Decode(new[] { 256 }));
            Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.Decode(new[] { -1 }));
        }

        [Fact]
        public void Settings_AreValidated()
        {
            Assert.Throws<ArgumentException>(() => CreateTokenizer(255));
            Assert.Throws<ArgumentException>(() => CreateTokenizer(256, "<pad>"));
            Assert.Throws<ArgumentException>(() => CreateTokenizer(300, "<pad>", "<pad>"));
            Assert.Throws<ArgumentException>(() => CreateTokenizer(300, ""));
            Assert.Throws<ArgumentNullException>(() => NetworkFactory.BuildBpeTokenizer(null));
        }
    }
}
