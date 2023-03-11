using System;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces.Neurons;

namespace ANN.Net.Neurons
{
    [Serializable]
    internal class InputNeuron : Neuron, IInputNeuron
    {
        public InputNeuron(ActivationTypes activationType, ushort recurrentInputs = 0, uint? id = null)
            : base(false, true, activationType, id)
        { }

        public override void Backpropagate(BackpropagateEventArgs errorSignal)
        {
            Outputs.AccumulateError(errorSignal);
        }

        public override void Propagate(NeuronPropagateEventArgs value)
        {
            ActivateNeuron(value);
        }
    }
}
