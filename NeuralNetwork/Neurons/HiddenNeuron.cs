using System;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces.Neurons;

namespace ANN.Net.Neurons
{
    [Serializable]
    internal class HiddenNeuron : Neuron, IHiddenNeuron
    {
        public HiddenNeuron(ActivationTypes activationType, uint? id = null)
            : base(true, true, activationType, id)
        { }

        public override void Backpropagate(BackpropagateEventArgs errorSignal)
        {
            Outputs.AccumulateError(errorSignal);
        }

        public override void Propagate(NeuronPropagateEventArgs value)
        {
            Inputs.AccountSignal(value);
        }
    }
}
