using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Utils;

namespace ANN.Net.ActivationFunctions
{
    internal class PReLUActivation : BaseActivation<PReLUActivation>, IActivationFunction
    {
        public Quad MinValue => 0;

        public Quad MaxValue => float.MaxValue;

        private Quad alpha;

        public PReLUActivation(Quad alpha)
        {
            this.alpha = alpha;
        }

        public Quad Activation(ref Quad x)
        {
            return QuadMath.Max(0, x) + QuadMath.Min(0, alpha * x);
        }

        public Quad Derivative(Quad x)
        {
            return QuadMath.Max(0, x) + QuadMath.Min(0, alpha * x);
        }
    }
}
