using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    [ActivationType(Type = ActivationTypes.GELU)]
    internal class GELUActivation : BaseActivation<GELUActivation>, IActivationFunction
    {
        public Quad MinValue => 0;

        public Quad MaxValue => float.MaxValue;

        private Quad alpha;

        public GELUActivation(Quad alpha)
        {
            this.alpha = alpha;
        }

        public Quad Activation(ref Quad x)
        {
            return (x / 2) * (1 + (Quad)Math.Tanh((Quad)Math.Sqrt(2 / Math.PI)) * (x + (alpha * (Quad)Math.Pow(x, 3))));
        }

        public Quad Derivative(Quad x)
        {

        }
    }
}
