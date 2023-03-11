using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using static System.Math;

namespace ANN.Net.Optimizers
{
    internal class RMSprop : Optimizer, IOptimizer
    {
        private Quad gradSquared = 0;
        private Quad learningRate = 0.001f;

        public RMSprop(float beta1 = 0.9F, float beta2 = 0.1F)
             : base(beta1, beta2)
        { }

        public Quad Calculate(Quad gradient, Quad weight)
        {
            gradSquared = beta1 * gradSquared + beta2 * (Quad)Pow(gradient, 2);
            return weight - learningRate / (Quad)Sqrt(gradSquared) * gradient;
        }

        /*
         * dw = compute_gradients(x, y)
         * grad_squared = 0.9 * grads_squared + 0.1 * dx * dx
         * w = w - (lr / np.sqrt(grad_squared)) * dw
         */
    }
}
