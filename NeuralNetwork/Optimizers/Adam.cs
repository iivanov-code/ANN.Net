using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using static System.Math;

namespace ANN.Net.Optimizers
{
    internal sealed class Adam : Optimizer, IOptimizer
    {
        private static Quad epsilon = 0;

        public Adam(float beta1 = 0.9F, float beta2 = 0.999F)
             : base(beta1, beta2)
        { }

        public Quad Calculate(Quad gradient, Quad weight)
        {
            m = beta1 * m + (1 - beta1) * gradient;
            v = beta2 * v + (1 - beta2) * (Quad)Pow(gradient, 2);
            Quad m_hat = m / (1 - (Quad)Pow(beta1, t));
            Quad v_hat = v / (1 - (Quad)Pow(beta2, t));
            t++;
            return weight - LearningRate * m_hat / ((Quad)Sqrt(v_hat) + epsilon);
        }

        /*
         * m = beta_1 * m + (1 - beta_1) * g
         * v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
         * m_hat = m / (1 - np.power(beta_1, t))
         * v_hat = v / (1 - np.power(beta_2, t))
         * w = w - step_size * m_hat / (np.sqrt(v_hat) + epsilon)
         */
    }
}
