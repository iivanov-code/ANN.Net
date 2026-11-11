using System;
using System.Text.Json.Serialization;

namespace ANN.Net.Remote.Abstactions.Models
{
    [Serializable]
    public class RemoteBackpropagationResult
    {
        [JsonPropertyName("loss")]
        public double Loss { get; set; }

        /// <summary>
        /// value - target for every output of the remote part
        /// </summary>
        [JsonPropertyName("es")]
        public double[] Errors { get; set; }

        /// <summary>
        /// Gradient of the loss with respect to every boundary neuron, keyed by synapse id
        /// </summary>
        [JsonPropertyName("gs")]
        public RemoteBackpropagationDTO[] Gradients { get; set; }
    }
}
