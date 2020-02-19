using NeuralNetwork.Enums;
using NeuralNetwork.HelperClasses;
using NeuralNetwork.Interfaces;
using System;

namespace NeuralNetwork.Neurons
{
    internal class InputNeuron : Neuron, IInputNeuron
    {
        public InputNeuron(ActivationTypes activationType, ushort recurrentInputs = 0)
            : base(activationType)
        {
            if (recurrentInputs != 0)
                this.Inputs = new SynapseCollection<ISynapse>(recurrentInputs);

            this.Outputs = new SynapseCollection<ISynapse>();
        }

        public override void Backpropagate(float errorSignal, float eWeightedSignal = 0, Action<float> updateWeight = null)
        {
            AccumulateError(errorSignal, eWeightedSignal, updateWeight);
            this.Outputs.ResetCounter();
            _error = 0;
        }

        public override void Propagate(float value)
        {
            _value = Function.Activation(value);

            foreach (var output in Outputs)
            {
                output.Propagate(_value);
            }

            _lastValue = _value;
            _value = 0;
        }
    }
}
