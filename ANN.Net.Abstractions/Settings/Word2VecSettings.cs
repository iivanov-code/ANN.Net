using System.ComponentModel;
using ANN.Net.Abstractions.Enums;

namespace ANN.Net.Abstractions.Settings
{
    public class Word2VecSettings
    {
        /// <summary>
        /// Number of distinct token ids; ids are in [0, VocabularySize)
        /// </summary>
        public int VocabularySize { get; set; }

        /// <summary>
        /// Size of the learned vector of every token
        /// </summary>
        [DefaultValue(100)]
        public ushort EmbeddingDimension { get; set; } = 100;

        [DefaultValue(Word2VecArchitecture.SkipGram)]
        public Word2VecArchitecture Architecture { get; set; }

        /// <summary>
        /// Number of tokens on each side of the center token that count as its context
        /// </summary>
        [DefaultValue(5)]
        public ushort WindowSize { get; set; } = 5;

        /// <summary>
        /// When true the window of every center token is shrunk to a random size in [1, WindowSize],
        /// which gives nearby tokens more weight
        /// </summary>
        [DefaultValue(true)]
        public bool UseDynamicWindow { get; set; } = true;

        /// <summary>
        /// Number of noise tokens scored against every real context token
        /// </summary>
        [DefaultValue(5)]
        public ushort NegativeSamplesCount { get; set; } = 5;

        /// <summary>
        /// Noise tokens are drawn with probability proportional to their count raised to this power
        /// </summary>
        [DefaultValue(0.75f)]
        public float NoiseDistributionPower { get; set; } = 0.75f;

        /// <summary>
        /// Tokens more frequent than this fraction of the corpus are randomly dropped from the
        /// training sequences; 0 disables subsampling
        /// </summary>
        [DefaultValue(1e-3f)]
        public float SubsamplingThreshold { get; set; } = 1e-3f;

        /// <summary>
        /// Seed of the random source used for initialization, window shrinking, subsampling
        /// and negative sampling; null gives a different sequence on every run
        /// </summary>
        public int? Seed { get; set; }

        [DefaultValue(OptimizerTypes.Normal)]
        public OptimizerTypes OptimizerType { get; set; }
    }
}
