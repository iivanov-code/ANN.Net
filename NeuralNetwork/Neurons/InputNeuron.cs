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

        /// <summary>
        /// Gradient of the loss with respect to the raw input value, from the last backpropagation
        /// </summary>
        public Quad Gradient => this.Delta;

        public override void Backpropagate(BackpropagateEventArgs errorSignal)
        {
            // the error of every output synapse is collected; once complete FeedbackError stores the gradient
            Outputs.AccumulateError(errorSignal);
        }

        public override void Propagate(NeuronPropagateEventArgs value)
        {
            ActivateNeuron(value);
        }
    }
}
