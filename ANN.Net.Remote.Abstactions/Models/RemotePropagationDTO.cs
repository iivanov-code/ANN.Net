using System;
using System.Linq;
using System.Text.Json.Serialization;
using System.Xml.Serialization;
using ANN.Net.Abstractions.Arguments;

namespace ANN.Net.Remote.Abstactions.Models
{
    [Serializable]
    [XmlRoot("RP")]
    public class RemotePropagationDTO
    {
        [XmlAttribute("id")]
        [JsonPropertyName("id")]
        public string Id { get; set; }

        [XmlAttribute("sa")]
        [JsonPropertyName("sa")]
        public bool ShouldActivate { get; set; }

        [XmlAttribute("cv")]
        [JsonPropertyName("cv")]
        public bool ClearValue { get; set; }

        [XmlElement("vs")]
        [JsonPropertyName("vs")]
        public double[] Values { get; set; }


        public static explicit operator RemotePropagationDTO(NeuronPropagateEventArgs eventArgs)
        {
            return new RemotePropagationDTO
            {
                ShouldActivate = eventArgs.ShouldActivate,
                ClearValue = eventArgs.ClearValue,
                Values = eventArgs.Values.Select(x => x.ToDouble(null)).ToArray()
            };
        }
    }
}
