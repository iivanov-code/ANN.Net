using System;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces.Neurons;

namespace ANN.Net.Neurons
{
    [Serializable]
    internal class OutputNeuron : Neuron, IOutputNeuron
    {
        private readonly bool shouldActivate;

        public OutputNeuron(ActivationTypes activationType, bool shouldActivate, uint? id = null)
            : base(true, true, activationType, id)
        {
            this.shouldActivate = shouldActivate;
        }

        protected override bool ShouldActivate => this.shouldActivate;

        /// <summary>
        /// Whether the activation function is applied to the output (it is not when the network applies soft-max)
        /// </summary>
        internal bool ApplyActivation => this.shouldActivate;

        public override void Backpropagate(BackpropagateEventArgs errorSignal)
        {
            FeedbackError(errorSignal);
        }

        public override void Propagate(NeuronPropagateEventArgs value)
        {
            Inputs.AccountSignal(value);
        }
    }
}
