using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.HelperClasses;
using System;

namespace ANN.Net.Neurons
{
    internal class OutputNeuron : Neuron, IOutputNeuron
    {
        public OutputNeuron(ActivationTypes activationType, ushort recurrentInputs = 0)
            : base(activationType)
        {
            Inputs = new SynapseCollection<ISynapse>(recurrentInputs);
        }

        public float Value => _lastValue;

        public override void Backpropagate(float errorSignal, float eWeightedSignal = 0, Action<float> updateWeight = null)
        {
            _error = errorSignal;
            _lastValue = Function.Prime(_lastValue);
            _error = _lastValue * _error; //derivative * errorSignal
            _lastValue = 0;
            foreach (ISynapse input in Inputs)
            {
                input.Backpropagate(_error);
            }
            _error = 0;
        }

        public override void Propagate(float value)
        {
            _value += value;
            Inputs.AccountSignal();
            if (Inputs.CheckCountAndReset())
            {
                _value = Function.Activation(_value);
                _lastValue = _value;
                _value = 0;
            }
        }
    }
}
