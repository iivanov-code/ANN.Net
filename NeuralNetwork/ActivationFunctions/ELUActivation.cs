using System;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    internal class ELUActivation : BaseActivation<ELUActivation>, IActivationFunction
    {
        public Quad MinValue => 0;

        public Quad MaxValue => float.MaxValue;

        private Quad alpha;

        public ELUActivation(Quad alpha)
        {
            this.alpha = alpha;
        }

        public Quad Activation(ref Quad x)
        {
            if (x > 0)
            {
                return x;
            }
            else
            {
                return alpha * ((Quad)Math.Exp(x) - 1);
            }
        }

        public Quad Derivative(Quad x)
        {
            if (x > 0)
            {
                return 1;
            }
            else
            {
                return alpha * (Quad)Math.Exp(x);
            }

        }
    }
}
