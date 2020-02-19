using NeuralNetwork.Enums;
using System.ComponentModel;

namespace NeuralNetwork.Settings
{
    public class HiddenLayerSettings
    {
        public ushort LayersCount { get; set; }
        public ushort NeuronsCount { get; set; }
        public bool HasBiasNeuron { get; set; }

        [DefaultValue(ActivationTypes.Sigmoid)]
        public ActivationTypes FunctionType { get; set; }
    }
}
