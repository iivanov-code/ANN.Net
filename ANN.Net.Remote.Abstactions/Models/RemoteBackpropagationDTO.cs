using System;
using System.Text.Json.Serialization;
using System.Xml.Serialization;
using ANN.Net.Abstractions.Arguments;

namespace ANN.Net.Remote.Abstactions.Models
{
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
            return new RemoteBackpropagationDTO
            {
                ErrorSignal = eventArgs.ErrorSignal,
                ErrorWeightedSignal = eventArgs.ErrorWeightedSignal,
                ShouldActivate = eventArgs.ShouldActivate
            };
        }
    }
}
