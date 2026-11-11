using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using static System.Math;

namespace ANN.Net.Optimizers
{
    /// <summary>
    /// AdaMax (Kingma and Ba, 2015): Adam with the infinity norm of past gradients instead of their squared average
    /// </summary>
    internal class AdaMax : Optimizer, IOptimizer
    {
        public AdaMax(float beta1 = 0.9F, float beta2 = 0.999F)
             : base(beta1, beta2)
        { }

        public Quad Calculate(Quad gradient, Quad weight)
        {
            m = (beta1 * m) + ((1 - beta1) * gradient);
            Quad mHat = m / (1 - (Quad)Pow(beta1, t));
            v = Max(beta2 * v, Abs(gradient));
            t++;
            return weight - (LearningRate * mHat / (v + Epsilon));
        }

        /*
         * m = beta_1 * m + (1 - beta_1) * g
         * m_hat = m / (1 - beta_1^t)
         * v = max(beta_2 * v, |g|)
         * w = w - step_size * m_hat / (v + epsilon)
         */
    }
}
