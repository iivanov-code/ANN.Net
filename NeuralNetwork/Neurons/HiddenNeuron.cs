using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.HelperClasses;
using System;

namespace ANN.Net.Neurons
{
    internal class HiddenNeuron : Neuron, IHiddenNeuron
    {
        public HiddenNeuron(ActivationTypes activationType, ushort recurrentInputs = 0)
            : base(activationType)
        {
            Inputs = new SynapseCollection<ISynapse>(recurrentInputs);
            Outputs = new SynapseCollection<ISynapse>();
        }

        public override void Backpropagate(float errorSignal, float eWeightedSignal = 0, Action<float> updateWeight = null)
        {
            AccumulateError(errorSignal, eWeightedSignal, updateWeight);
            if (Outputs.CheckCountAndReset())
            {
                _lastValue = Function.Prime(_lastValue);
                _error = _lastValue * _error; //derivative * errorSignal
                _lastValue = 0;
                foreach (ISynapse input in Inputs)
                {
                    input.Backpropagate(_error);
                }
                _error = 0;
            }
        }

        public override void Propagate(float value)
        {
            _value += value;
            Inputs.AccountSignal();
            if (Inputs.CheckCountAndReset())
            {
                _value = Function.Activation(_value);
                foreach (var output in Outputs)
                {
                    output.Propagate(_value);
                }
                _lastValue = _value;
                _value = 0;
            }
        }
    }
}
