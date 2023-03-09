using ANN.Net.Abstractions.Interfaces;
using ANN.Net.HelperClasses;
using System;

namespace ANN.Net.Connections
{
    internal class RecurrentSynapse : ISynapse
    {
#if DEBUG
        public INeuron Input { get; }
        public INeuron Output { get; }
#endif

        public float Weight { get; private set; }
        public float InputWeight { get; private set; }
        public float ForgetWeight { get; private set; }
        public float OutputWeight { get; private set; }
        public Rate LearningRate { get; set; } = 1;
        public Action<float, float, Action<float>> BackpropagateConnection { get; protected set; }
        public Action<float> PropagateConnection { get; protected set; }
        public Action<float> ForgetGateConnection { get; protected set; }
        public Action<float> OutputGateConnection { get; protected set; }
        public Action<float> InputGateConnection { get; protected set; }

        public RecurrentSynapse(INeuron input, ILSTMCell output, float weight)
        {
            BackpropagateConnection = input.Backpropagate;
            PropagateConnection = output.Propagate;
            ForgetGateConnection = output.ForgetGate;
            OutputGateConnection = output.OutputGate;
            InputGateConnection = output.InputGate;
            Weight = weight;
        }

        public void Backpropagate(float signal)
        {
            BackpropagateConnection(signal, signal * Weight, UpdateWeight);
        }

        public void UpdateWeight(float gradient)
        {
            float delta = gradient * LearningRate;
            Weight -= delta;
        }

        public void Propagate(float value)
        {
            PropagateConnection(Weight * value);
        }
    }
}
