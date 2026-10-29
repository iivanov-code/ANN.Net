using System;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Neurons;

namespace ANN.Net.Remote
{
    /// <summary>
    /// Stand-in for a neuron that lives in another part: an identity input neuron whose value is set from a
    /// message and which reports when its gradient, the sum over its local consumers, is complete
    /// </summary>
    [Serializable]
    internal sealed class GhostNeuron : InputNeuron
    {
        private readonly Action<GhostNeuron> onGradient;

        public GhostNeuron(int index, Action<GhostNeuron> onGradient)
            : base(ActivationTypes.Identity)
        {
            this.Index = index;
            this.onGradient = onGradient;
        }

        /// <summary>
        /// Position of the real neuron in its layer
        /// </summary>
        public int Index { get; }

        protected override void FeedbackError(BackpropagateEventArgs errorSignal)
        {
            base.FeedbackError(errorSignal);
            this.onGradient?.Invoke(this);
        }
    }
}
