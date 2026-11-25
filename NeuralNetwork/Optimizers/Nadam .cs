using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using static System.Math;

namespace ANN.Net.Optimizers
{
    /// <summary>
    /// Nadam (Dozat, 2016): Adam with Nesterov momentum, which looks one momentum step ahead. The momentum is the
    /// constant beta_1, and the look-ahead momentum term is corrected with 1 - beta_1^(t+1). PyTorch and Keras schedule
    /// the momentum by default (beta_1 * (1 - 0.5 * 0.96^(t / 250)), half of beta_1 at the start); this is their rule
    /// without that schedule
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
            Quad nextCorrection = 1 - (Quad)Pow(beta1, t + 1);
            Quad mHat = (beta1 * m / nextCorrection) + ((1 - beta1) * gradient / correction);
            Quad vHat = v / (1 - (Quad)Pow(beta2, t));
            t++;
            return weight - (LearningRate * mHat / ((Quad)Sqrt(vHat) + Epsilon));
        }

        /*
         * m = beta_1 * m + (1 - beta_1) * g
         * v = beta_2 * v + (1 - beta_2) * g^2
         * m_hat = beta_1 * m / (1 - beta_1^(t+1)) + (1 - beta_1) * g / (1 - beta_1^t)
         * v_hat = v / (1 - beta_2^t)
         * w = w - step_size * m_hat / (sqrt(v_hat) + epsilon)
         */
    }
}
