using System;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces.Neurons;

namespace ANN.Net.Neurons
{
    /// <summary>
    /// Neuron with the constant output 1; the weights of its output synapses are the biases of the next layer
    /// </summary>
    [Serializable]
    internal class BiasNeuron : Neuron, IInputNeuron, IHiddenNeuron
    {
        public BiasNeuron(ActivationTypes activationType)
             : base(false, true, activationType)
        {
            this.Value = 1;
            this.PreActivation = 1;
        }

        public Quad Gradient => 0;

        public override void Backpropagate(BackpropagateEventArgs errorSignal)
        {
            Outputs.AccumulateError(errorSignal);
        }

        public override void Propagate(NeuronPropagateEventArgs value)
        {
            ActivateNeuron(value);
        }

        protected override void ActivateNeuron(NeuronPropagateEventArgs value)
        {
            this.WasActivated = false;
            this.Outputs.PropagateForEach(new NeuronPropagateEventArgs(1));
        }

        protected override void FeedbackError(BackpropagateEventArgs value)
        {
            // the output is constant, there is nothing to pass back
        }
    }
}
