using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;

namespace ANN.Net.Optimizers
{
    /// <summary>
    /// Nesterov momentum in the form of Sutskever et al. (2013) used by PyTorch: the step adds the momentum once more,
    /// which approximates the gradient one momentum step ahead
    /// </summary>
    internal class Nesterov : LearningRateOptimizer, IOptimizer
    {
        private readonly Quad beta = 0.9f;
        private Quad v = 0;

        public Quad Calculate(Quad gradient, Quad weight)
        {
            v = (beta * v) + gradient;
            return weight - (LearningRate * (gradient + (beta * v)));
        }

        /*
         * v = beta * v + g
         * w = w - step_size * (g + beta * v)
         */
    }
}
