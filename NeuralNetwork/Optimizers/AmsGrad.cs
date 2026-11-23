using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using static System.Math;

namespace ANN.Net.Optimizers
{
    /// <summary>
    /// AMSGrad (Reddi, Kale and Kumar, 2018): Adam that divides by the largest squared average seen so far, so the
    /// step size never grows again. Bias-corrected like Adam, as in PyTorch's <c>Adam(amsgrad=True)</c>; the paper
    /// leaves the correction out of its pseudo-code, which makes the first steps several times the learning rate
    /// </summary>
    internal sealed class AmsGrad : Optimizer, IOptimizer
    {
        private Quad vHat = 0;

        public AmsGrad(float beta1 = 0.9F, float beta2 = 0.999F)
             : base(beta1, beta2)
        { }

        public Quad Calculate(Quad gradient, Quad weight)
        {
            m = (beta1 * m) + ((1 - beta1) * gradient);
            v = (beta2 * v) + ((1 - beta2) * gradient * gradient);
            vHat = Max(v, vHat);
            Quad mCorrected = m / (1 - (Quad)Pow(beta1, t));
            Quad vCorrected = vHat / (1 - (Quad)Pow(beta2, t));
            t++;
            return weight - (LearningRate * mCorrected / ((Quad)Sqrt(vCorrected) + Epsilon));
        }

        /* m = beta_1 * m + (1 - beta_1) * g
         * v = beta_2 * v + (1 - beta_2) * g^2
         * v_hat = max(v, v_hat)
         * w = w - step_size * (m / (1 - beta_1^t)) / (sqrt(v_hat / (1 - beta_2^t)) + epsilon)
         */
    }
}
