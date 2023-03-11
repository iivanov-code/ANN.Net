using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;

namespace ANN.Net.Neurons
{
    internal class MultiDimentionalNeuron : Neuron, INeuron, IHiddenNeuron, IInputNeuron
    {
        private INetwork network;

        public MultiDimentionalNeuron(INetwork network, ActivationTypes activationType)
             : base(true, true, activationType)
        {
            this.network = network;
        }

        public override void Backpropagate(BackpropagateEventArgs errorSignal)
        {
            this.Outputs.AccumulateError(errorSignal);
        }

        public override void Propagate(NeuronPropagateEventArgs value)
        {
            this.Inputs.AccountSignal(value);
        }

        protected override void ActivateNeuron(NeuronPropagateEventArgs value)
        {
            if (value.ShouldActivate)
            {
                Quad activationValue = value.Value;
                activationValue = Function.Activation(ref activationValue);
                value.Value = activationValue;
            }

            value.Value = this.network.Propagate(value.Value)[0];
            this.Outputs.PropagateForEach(value);
        }

        protected override void FeedbackError(BackpropagateEventArgs errorSignal)
        {
            if (errorSignal.ShouldActivate)
                Outputs.Value = Function.Derivative(Outputs.Value);

            errorSignal.ErrorSignal = Outputs.Value * Inputs.Value; //derivative * errorSignal
            errorSignal.ErrorSignal = this.network.Backpropagate(errorSignal.ErrorSignal);
            Inputs.BackpropagateForEach(errorSignal);
            Inputs.ClearValue();
            Outputs.ClearValue();
        }
    }
}
