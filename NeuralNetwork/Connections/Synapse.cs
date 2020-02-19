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
        public float Weight
        {
            get
            {
                return weight;
            }
        }
#endif

        private float weight;
        public Rate LearningRate { get; set; } = 1;
        public Action<float, float, Action<float>> BackpropagateConnection { get; protected set; }
        public Action<float> PropagateConnection { get; protected set; }

        public Synapse(INeuron input, INeuron output, float weight)
        {
            BackpropagateConnection = input.Backpropagate;
            PropagateConnection = output.Propagate;
            this.weight = weight;
        }

        public void Backpropagate(float signal)
        {
            BackpropagateConnection(signal, signal * weight, this.UpdateWeight);
        }

        public void UpdateWeight(float gradient)
        {
            float delta = gradient * LearningRate;
            weight -= delta;
        }

        public void Propagate(float value)
        {
            PropagateConnection(weight * value);
        }
    }
}
