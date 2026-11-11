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

        public Quad Gradient => this.Delta;

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
            this.PreActivation = value.Value;
            this.WasActivated = value.ShouldActivate;

            Quad output = value.Value;
            if (this.WasActivated)
            {
                output = this.Function.Activation(ref output);
            }

            output = this.network.Propagate(output)[0];
            this.Value = output;
            this.Outputs.PropagateForEach(new NeuronPropagateEventArgs(output));
        }

        protected override void FeedbackError(BackpropagateEventArgs errorSignal)
        {
            Quad delta = this.network.Backpropagate(this.ToDelta(errorSignal));
            this.Delta = delta;
            this.Inputs.BackpropagateForEach(new BackpropagateEventArgs(delta));
        }
    }
}
