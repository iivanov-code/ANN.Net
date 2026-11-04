namespace ANN.Net.Abstractions.Optimizers
{
    public abstract class Optimizer : LearningRateOptimizer
    {
        public Optimizer(float beta1 = 0.9f, float beta2 = 0.999f)
        {
            this.beta1 = beta1;
            this.beta2 = beta2;
        }

        /// <summary>
        /// Keeps adaptive optimizers from dividing by zero when the gradients so far were zero
        /// </summary>
        protected const float Epsilon = 1e-8f;

        protected Quad beta1, beta2;
        protected Quad m = 0;
        protected Quad v = 0;
        protected uint t = 1;
    }
}
