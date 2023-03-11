using System;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Utils;

namespace ANN.Net.Neurons
{
    [Serializable]
    internal abstract class Neuron : BaseNeuron, INeuron
    {
        public Neuron(bool initInputs, bool initOutputs, ActivationTypes activationType, uint? id = null)
             : base(initInputs, initOutputs, id)
        {
            this.Function = NetworkUtils.GetActivation(activationType);
        }

        public IActivationFunction Function { get; private set; }

        protected override void ActivateNeuron(NeuronPropagateEventArgs value)
        {
            if (value.ShouldActivate)
            {
                Quad activationValue = value.Value;
                activationValue = Function.Activation(ref activationValue);
                value.Value = activationValue;
            }

            this.Outputs.PropagateForEach(value);
        }

        protected override void FeedbackError(BackpropagateEventArgs errorSignal)
        {
            if (errorSignal.ShouldActivate)
                Outputs.Value = Function.Derivative(Outputs.Value);

            errorSignal.ErrorSignal = Outputs.Value * Inputs.Value; //derivative * errorSignal
            Inputs.BackpropagateForEach(errorSignal);
            Inputs.ClearValue();
            Outputs.ClearValue();
        }
    }
}
