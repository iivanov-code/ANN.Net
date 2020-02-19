using NeuralNetwork.HelperClasses;
using NeuralNetwork.Interfaces;
using System;

namespace NeuralNetwork.Connections
{
    [Serializable]
    internal class Synapse : ISynapse
    {
#if DEBUG
        public INeuron Input { get; }
        public INeuron Output { get; }
        public float Weight { get; private set; }
#endif
        public Rate LearningRate { get; set; } = 1;
        public Action<float, float, Action<float>> BackpropagateConnection { get; protected set; }
        public Action<float> PropagateConnection { get; protected set; }

        public Synapse(INeuron input, INeuron output, float weight)
        {
            BackpropagateConnection = input.Backpropagate;
            PropagateConnection = output.Propagate;
            Weight = weight;
        }

        public void Backpropagate(float signal)
        {
            BackpropagateConnection(signal, signal * Weight, this.UpdateWeight);
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
