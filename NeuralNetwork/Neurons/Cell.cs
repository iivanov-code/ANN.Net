using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Settings;

namespace ANN.Net.Neurons
{
    public abstract class Cell
    {
        protected Quad[] cellState;

        public uint ID { get; set; }

        public ISynapses Outputs { get; protected set; }

        public ISynapses Inputs { get; protected set; }

        protected Cell(ref ushort inputNeurons, ushort outputNeurons)
        {
            inputNeurons += outputNeurons;
        }

        protected static INetwork BuildGate(ushort inputNeurons, ushort outputNeurons, ActivationTypes activation = ActivationTypes.Sigmoid)
        {
            return NetworkFactory.BuildFFN(new NetworkSettings
            {
                InputSettings = new InputLayerSettings
                {
                    HasBiasNeuron = false,
                    ActivationType = activation,
                    NeuronsCount = inputNeurons,
                    InitializationType = WeightInitType.Xavier
                },
                OutputSettings = new OutputLayerSettings
                {
                    HasBiasNeuron = false,
                    ActivationType = activation,
                    NeuronsCount = outputNeurons,
                    InitializationType = WeightInitType.Xavier
                }
            }, null);
        }
    }
}
