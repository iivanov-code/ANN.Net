using System;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;

namespace ANN.Net.Optimizers
{
    [Serializable]
    internal class Normal : LearningRateOptimizer, IOptimizer
    {
        public Quad Calculate(Quad gradient, Quad weight)
        {
            Quad delta = gradient * LearningRate;
            weight -= delta;
            return weight;
        }
    }
}
