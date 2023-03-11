using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;

namespace ANN.Net.Optimizers
{
    internal class Momentum : LearningRateOptimizer, IOptimizer
    {
        private Quad beta1 = 0.9f;
        private Quad v = 0;

        public Quad Calculate(Quad gradient, Quad weight)
        {
            v = beta1 * v + LearningRate * gradient;
            return weight - v;
        }
    }
}
