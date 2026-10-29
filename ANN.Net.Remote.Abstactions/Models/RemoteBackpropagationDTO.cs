using System;
using System.Text.Json.Serialization;
using System.Xml.Serialization;
using ANN.Net.Abstractions.Arguments;

namespace ANN.Net.Remote.Abstactions.Models
{
    /// <summary>
    /// Gradient of the remote part with respect to the value one remote synapse carried
    /// </summary>
    [Serializable]
    [XmlRoot("RBP")]
    public class RemoteBackpropagationDTO
    {
        [XmlAttribute("id")]
        [JsonPropertyName("id")]
        public string Id { get; set; }

        [XmlAttribute("es")]
        [JsonPropertyName("es")]
        public double ErrorSignal { get; set; }

        [XmlAttribute("ews")]
        [JsonPropertyName("ews")]
        public float ErrorWeightedSignal { get; set; }

        [XmlAttribute("sa")]
        [JsonPropertyName("sa")]
        public bool ShouldActivate { get; set; }

        public static explicit operator RemoteBackpropagationDTO(BackpropagateEventArgs eventArgs)
        {
            if (eventArgs == null) throw new ArgumentNullException(nameof(eventArgs));

            return new RemoteBackpropagationDTO
            {
                ErrorSignal = eventArgs.ErrorSignal,
                ErrorWeightedSignal = eventArgs.ErrorWeightedSignal,
                ShouldActivate = eventArgs.ShouldActivate
            };
        }

        public BackpropagateEventArgs ToEventArgs()
        {
            Quad gradient = (Quad)this.ErrorSignal;
            return new BackpropagateEventArgs(gradient, gradient, null)
            {
                ShouldActivate = this.ShouldActivate
            };
        }
    }
}
