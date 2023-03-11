using System.ComponentModel;
using ANN.Net.Abstractions.Enums;

namespace ANN.Net.Abstractions.Settings
{
    public class CommonLayerSettings
    {
        public bool HasBiasNeuron { get; set; }

        [DefaultValue(WeightInitType.Xavier)]
        public WeightInitType InitializationType { get; set; }

        [DefaultValue(OptimizerTypes.Normal)]
        public OptimizerTypes OptimizerType { get; set; }

        public ushort NeuronsCount { get; set; }

        [DefaultValue(ActivationTypes.Sigmoid)]
        public ActivationTypes ActivationType { get; set; }
    }
}
