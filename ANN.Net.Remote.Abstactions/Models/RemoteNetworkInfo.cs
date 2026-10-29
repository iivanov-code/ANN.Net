using System;
using System.Text.Json.Serialization;

namespace ANN.Net.Remote.Abstactions.Models
{
    [Serializable]
    public class RemoteNetworkInfo
    {
        /// <summary>
        /// Boundary neurons the part expects, bias neurons not counted
        /// </summary>
        [JsonPropertyName("in")]
        public int InputCount { get; set; }

        [JsonPropertyName("out")]
        public int OutputCount { get; set; }
    }
}
