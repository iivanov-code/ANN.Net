using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    [Serializable]
    [ActivationType(Type = ActivationTypes.Gaussian)]
    internal class GaussianActivation : BaseActivation<GaussianActivation>, IActivationFunction
    {
        public Quad MinValue => 0;

        public Quad MaxValue => 1;

        public Quad Activation(ref Quad x)
        {
            return (Quad)Math.Exp(-(double)x * x);
        }

        public Quad Derivative(Quad x)
        {
            double value = x;
            return (Quad)(-2 * value * Math.Exp(-value * value));
        }
    }
}
