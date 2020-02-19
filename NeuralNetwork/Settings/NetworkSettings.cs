using NeuralNetwork.Enums;
using System.ComponentModel;

namespace NeuralNetwork.Settings
{
    public class NetworkSettings
    {
        public float MinNetworkValue { get; set; }
        public float MaxNetworkValue { get; set; }
        public bool ShouldNormalize { get; set; } = true;
        public bool HasBiasNeuron { get; set; }

        public ushort InputNeuronsCount { get; set; }
        [DefaultValue(ActivationTypes.Sigmoid)]
        public ActivationTypes InputLayerFunction { get; set; }


        public ushort OutputNeuronsCount { get; set; }

        [DefaultValue(ActivationTypes.Sigmoid)]
        public ActivationTypes OutputLayerFunction { get; set; }
    }
}
