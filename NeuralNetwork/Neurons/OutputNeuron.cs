using System;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces.Neurons;

namespace ANN.Net.Neurons
{
    [Serializable]
    internal class OutputNeuron : Neuron, IOutputNeuron
    {
        private bool shouldActivate;

        public OutputNeuron(ActivationTypes activationType, bool shouldActivate, uint? id = null)
            : base(true, true, activationType, id)
        {
            this.shouldActivate = shouldActivate;
        }

        public Quad Value => this.Outputs.Value;

        public override void Backpropagate(BackpropagateEventArgs errorSignal)
        {
            FeedbackError(errorSignal);
        }

        public override void Propagate(NeuronPropagateEventArgs value)
        {
            Outputs.AccountSignal(value);
        }

        protected override void ActivateNeuron(NeuronPropagateEventArgs value)
        {
            base.ActivateNeuron(value);
        }

        protected override void FeedbackError(BackpropagateEventArgs value)
        {
            base.FeedbackError(value);
        }
    }
}
