using System.ComponentModel;

namespace ANN.Net.Abstractions.Settings
{
    public class LanguageModelSettings
    {
        /// <summary>
        /// Number of tokens the model sees at once. Training windows and the generation context
        /// are cut to it; it must not exceed the network's maximum sequence length
        /// </summary>
        public ushort ContextLength { get; set; }

        /// <summary>
        /// Distance between the starts of consecutive training windows; 0 means ContextLength,
        /// which makes consecutive windows overlap by one token so every token is predicted once
        /// </summary>
        [DefaultValue(0)]
        public ushort WindowStride { get; set; }

        /// <summary>
        /// Name of the special token put in front of every sequence; null or empty for none
        /// </summary>
        [DefaultValue("<bos>")]
        public string BeginToken { get; set; } = "<bos>";

        /// <summary>
        /// Name of the special token appended to every sequence and used to stop generation; null or empty for none
        /// </summary>
        [DefaultValue("<eos>")]
        public string EndToken { get; set; } = "<eos>";

        /// <summary>
        /// Visit the training windows in a random order every epoch
        /// </summary>
        [DefaultValue(true)]
        public bool ShuffleWindows { get; set; } = true;

        /// <summary>
        /// Seed of the random source used for shuffling and sampling; null gives a different sequence on every run
        /// </summary>
        public int? Seed { get; set; }
    }
}
