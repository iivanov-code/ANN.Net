using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using static System.Math;

namespace ANN.Net.Optimizers
{
    internal sealed class AmsGrad : Optimizer, IOptimizer
    {
        private Quad vHat = 0;
        private Quad epsilon = 0;

        public Quad Calculate(Quad gradient, Quad weight)
        {
            m = beta1 * m + (1 - beta1) * gradient;
            v = beta2 * v + (1 - beta2) * (Quad)Pow(gradient, 2);
            vHat = Max(v, vHat);
            return weight - LearningRate * m / ((Quad)Sqrt(vHat) + epsilon);
        }

        /* g = compute_gradient(x, y)
         * m = beta_1 * m + (1 - beta_1) * g
         * v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
         * v_hat = np.maximum(v, v_hat)
         * w = w - step_size * m / (np.sqrt(v_hat) + epsilon)
         */
    }
}
