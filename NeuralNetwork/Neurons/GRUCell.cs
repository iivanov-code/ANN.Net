using NeuralNetwork.Enums;
using NeuralNetwork.Interfaces;
using System;

namespace NeuralNetwork.Neurons
{
    internal class GRUCell : Neuron, INeuron
    {
        public GRUCell(ActivationTypes activationType)
            : base(activationType)
        { }

        public override void Backpropagate(float errorSignal, float eWeightedSignal = 0, Action<float> updateWeight = null)
        {
            throw new NotImplementedException();
        }

        public override void Propagate(float value)
        {
            throw new NotImplementedException();
        }
    }
}
