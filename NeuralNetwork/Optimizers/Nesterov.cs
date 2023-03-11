using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;

namespace ANN.Net.Optimizers
{
    internal class Nesterov : LearningRateOptimizer, IOptimizer
    {
        private Quad v = 0;
        private Quad beta = 0.9f;

        public Quad Calculate(Quad gradient, Quad weight)
        {
            v = v * beta + LearningRate * gradient * (weight - beta * v);
            return weight - v;
        }
    }
}
