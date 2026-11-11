using System;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Utils;

namespace ANN.Net.Neurons
{
    /// <summary>
    /// Neuron that sums its weighted inputs, activates the sum and sends the result along every output synapse.
    /// It remembers the sum before activation so the error can be turned into a gradient when it comes back
    /// </summary>
    [Serializable]
    internal abstract class Neuron : BaseNeuron, INeuron
    {
        public Neuron(bool initInputs, bool initOutputs, ActivationTypes activationType, uint? id = null)
             : base(initInputs, initOutputs, id)
        {
            this.Function = NetworkUtils.GetActivation(activationType);
        }

        public IActivationFunction Function { get; private set; }

        /// <summary>
        /// Weighted input sum of the last propagation, before activation
        /// </summary>
        public Quad PreActivation { get; protected set; }

        /// <summary>
        /// Gradient of the loss with respect to <see cref="PreActivation"/> from the last backpropagation
        /// </summary>
        public Quad Delta { get; protected set; }

        /// <summary>
        /// Whether the activation function was applied during the last propagation
        /// </summary>
        protected bool WasActivated { get; set; }

        /// <summary>
        /// Whether this neuron applies its activation function at all
        /// </summary>
        protected virtual bool ShouldActivate => true;

        protected override void ActivateNeuron(NeuronPropagateEventArgs value)
        {
            this.PreActivation = value.Value;
            this.WasActivated = value.ShouldActivate && this.ShouldActivate;

            Quad output = value.Value;
            if (this.WasActivated)
            {
                output = this.Function.Activation(ref output);
            }

            this.Value = output;

            // every synapse weights its own copy of the message, so one synapse cannot change what the next one sees
            this.Outputs.PropagateForEach(new NeuronPropagateEventArgs(output));
        }

        protected override void FeedbackError(BackpropagateEventArgs errorSignal)
        {
            this.Delta = this.ToDelta(errorSignal);

            if (this.Inputs != null)
            {
                this.Inputs.BackpropagateForEach(new BackpropagateEventArgs(this.Delta));
            }
        }

        /// <summary>
        /// Turns the gradient with respect to the output into the gradient with respect to the pre-activation sum
        /// </summary>
        protected Quad ToDelta(BackpropagateEventArgs errorSignal)
        {
            Quad delta = errorSignal.ErrorSignal;
            if (errorSignal.ShouldActivate && this.WasActivated)
            {
                delta *= this.Function.Derivative(this.PreActivation);
            }

            return delta;
        }
    }
}
