using System;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;

namespace ANN.Net.Neurons
{
    internal class GRUCell : Cell, IGRUCell
    {
        private INetwork resetGate, updateGate, targetGate;

        public GRUCell(ref ushort inputNeurons, ushort outputNeurons)
            : base(ref inputNeurons, outputNeurons)
        {
            this.resetGate = BuildGate(inputNeurons, outputNeurons);
            this.updateGate = BuildGate(inputNeurons, outputNeurons);
            this.targetGate = BuildGate(inputNeurons, outputNeurons, ActivationTypes.HyperbolicTangens);
        }

        public void Backpropagate(BackpropagateEventArgs errorSignal)
        {
            throw new NotImplementedException();
        }

        public void Propagate(NeuronPropagateEventArgs value)
        {
            Quad[] inputResult = this.resetGate.Propagate(value.Values);
            Quad[] updateResult = this.updateGate.Propagate(value.Values);
            Quad[] targetResult = this.targetGate.Propagate(value.Values);

        }
    }
}
