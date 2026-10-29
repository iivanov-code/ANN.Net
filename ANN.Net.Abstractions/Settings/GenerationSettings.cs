using System.ComponentModel;

namespace ANN.Net.Abstractions.Settings
{
    public class GenerationSettings
    {
        /// <summary>
        /// Most tokens generated after the prompt
        /// </summary>
        [DefaultValue(64)]
        public ushort MaxNewTokens { get; set; } = 64;

        /// <summary>
        /// 0 picks the most likely token every time; higher values flatten the distribution before sampling
        /// </summary>
        [DefaultValue(1f)]
        public float Temperature { get; set; } = 1f;

        /// <summary>
        /// Sampling only considers this many most likely tokens; 0 considers all of them
        /// </summary>
        [DefaultValue(0)]
        public ushort TopK { get; set; }

        /// <summary>
        /// Stop as soon as the end token is produced
        /// </summary>
        [DefaultValue(true)]
        public bool StopAtEndToken { get; set; } = true;
    }
}
