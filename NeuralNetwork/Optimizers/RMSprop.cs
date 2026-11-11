using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using static System.Math;

namespace ANN.Net.Optimizers
{
    /// <summary>
    /// RMSprop (Hinton): divides the step by a moving average of the squared gradient
    /// </summary>
    internal class RMSprop : Optimizer, IOptimizer
    {
        private Quad gradSquared = 0;

        /// <param name="beta1">Decay of the moving average (rho)</param>
        /// <param name="beta2">Not used; kept for compatibility</param>
        public RMSprop(float beta1 = 0.9F, float beta2 = 0.1F)
             : base(beta1, beta2)
        { }

        public Quad Calculate(Quad gradient, Quad weight)
        {
            gradSquared = (beta1 * gradSquared) + ((1 - beta1) * gradient * gradient);
            return weight - (LearningRate * gradient / ((Quad)Sqrt(gradSquared) + Epsilon));
        }

        /*
         * grad_squared = rho * grad_squared + (1 - rho) * g^2
         * w = w - step_size * g / (sqrt(grad_squared) + epsilon)
         */
    }
}
