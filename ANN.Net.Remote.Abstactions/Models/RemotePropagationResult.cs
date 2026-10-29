using System;
using System.Text.Json.Serialization;

namespace ANN.Net.Remote.Abstactions.Models
{
    [Serializable]
    public class RemotePropagationResult
    {
        /// <summary>
        /// Outputs of the remote part
        /// </summary>
        [JsonPropertyName("vs")]
        public double[] Values { get; set; }
    }
}
