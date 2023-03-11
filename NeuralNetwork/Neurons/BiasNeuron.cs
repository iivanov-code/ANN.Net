using System;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces.Neurons;

namespace ANN.Net.Neurons
{
    [Serializable]
    internal class BiasNeuron : Neuron, IInputNeuron, IHiddenNeuron
    {
        public BiasNeuron(ActivationTypes activationType)
             : base(true, true, activationType)
        { }

        public override void Backpropagate(BackpropagateEventArgs errorSignal)
        {
            Outputs.AccumulateError(errorSignal);
        }

        protected override void ActivateNeuron(NeuronPropagateEventArgs value)
        {
            this.Outputs.PropagateForEach(new NeuronPropagateEventArgs(Function.MaxValue));
        }

        protected override void FeedbackError(BackpropagateEventArgs value)
        {
            //Should not propagate it's error because it's value is fixed: 1
            //this.Inputs.BackpropagateForEach(value);
        }

        public override void Propagate(NeuronPropagateEventArgs value)
        {
            Inputs.AccountSignal(value);
        }
    }
}
