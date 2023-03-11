using System;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces.Neurons;

namespace ANN.Net.Neurons
{
    internal class GRUCell : Neuron, IHiddenNeuron
    {
        public GRUCell(ActivationTypes activationType)
            : base(true, true, activationType)
        { }

        public override void Backpropagate(BackpropagateEventArgs errorSignal)
        {
            throw new NotImplementedException();
        }

        public override void Propagate(NeuronPropagateEventArgs value)
        {
            throw new NotImplementedException();
        }
    }
}
