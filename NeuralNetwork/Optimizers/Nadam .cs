using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using static System.Math;

namespace ANN.Net.Optimizers
{
    /// <summary>
    /// Nadam (Dozat, 2016): Adam with Nesterov momentum, which looks one momentum step ahead
    /// </summary>
    internal sealed class Nadam : Optimizer, IOptimizer
    {
        public Nadam(float beta1 = 0.9F, float beta2 = 0.999F)
             : base(beta1, beta2)
        { }

        public Quad Calculate(Quad gradient, Quad weight)
        {
            m = (beta1 * m) + ((1 - beta1) * gradient);
            v = (beta2 * v) + ((1 - beta2) * gradient * gradient);
            Quad correction = 1 - (Quad)Pow(beta1, t);
            Quad mHat = (beta1 * m / correction) + ((1 - beta1) * gradient / correction);
            Quad vHat = v / (1 - (Quad)Pow(beta2, t));
            t++;
            return weight - (LearningRate * mHat / ((Quad)Sqrt(vHat) + Epsilon));
        }

        /*
         * m = beta_1 * m + (1 - beta_1) * g
         * v = beta_2 * v + (1 - beta_2) * g^2
         * m_hat = beta_1 * m / (1 - beta_1^t) + (1 - beta_1) * g / (1 - beta_1^t)
         * v_hat = v / (1 - beta_2^t)
         * w = w - step_size * m_hat / (sqrt(v_hat) + epsilon)
         */
    }
}
