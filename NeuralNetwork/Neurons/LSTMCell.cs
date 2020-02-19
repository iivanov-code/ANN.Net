using NeuralNetwork.ActivationFunctions;
using NeuralNetwork.Enums;
using NeuralNetwork.HelperClasses;
using NeuralNetwork.Interfaces;
using System;

namespace NeuralNetwork.Neurons
{
    internal class LSTMCell : HiddenNeuron, IHiddenNeuron, ILSTMCell
    {
        public LSTMCell(ActivationTypes activationType, ushort recurrentInputs = 0)
            : base(activationType, recurrentInputs)
        {
            this.InputGateSynapses = new SynapseCollection<ISynapse>(recurrentInputs);
            this.ForgetGateSynapses = new SynapseCollection<ISynapse>(recurrentInputs);
            this.OutputGateSypanses = new SynapseCollection<ISynapse>(recurrentInputs);
        }

        public ISynapseCollection<ISynapse> ForgetGateSynapses { get; }
        public ISynapseCollection<ISynapse> InputGateSynapses { get; }
        public ISynapseCollection<ISynapse> OutputGateSypanses { get; }

        private float cellState = 0;
        private float inputGateValue = 0;
        private float forgetGateValue = 0;
        private float outputGateValue = 0;
        public override void Backpropagate(float errorSignal, float eWeightedSignal = 0, Action<float> updateWeight = null)
        {
            base.Backpropagate(0, 0, null);
        }

        public override void Propagate(float value)
        {
            _value += value;
            Inputs.AccountSignal();
            if (Inputs.CheckCountAndReset())
            {
                _value = TanHActivation.Activate(_value);
                Calculate();
            }
        }

        public void ForgetGate(float value)
        {
            forgetGateValue += value;
            ForgetGateSynapses.AccountSignal();
            if (ForgetGateSynapses.CheckCountAndReset())
            {
                forgetGateValue = SigmoidActivation.Activate(forgetGateValue + cellState);
                Calculate();
            }
        }

        public void InputGate(float value)
        {
            inputGateValue += value;
            InputGateSynapses.AccountSignal();
            if (InputGateSynapses.CheckCountAndReset())
            {
                inputGateValue = SigmoidActivation.Activate(inputGateValue + cellState);
                Calculate();
            }
        }

        public void OutputGate(float value)
        {
            outputGateValue += value;
            OutputGateSypanses.AccountSignal();
            if (OutputGateSypanses.CheckCountAndReset())
            {
                Calculate();
            }
        }

        private void Calculate()
        {
            if (Inputs.Zeroed() && InputGateSynapses.Zeroed() && ForgetGateSynapses.Zeroed() && OutputGateSypanses.Zeroed())
            {
                cellState = (inputGateValue * _value) + (forgetGateValue * cellState);
                outputGateValue = SigmoidActivation.Activate(outputGateValue + cellState);
                _value = TanHActivation.Activate(cellState) * outputGateValue;
                _lastValue = _value;
            }
        }

    }
}