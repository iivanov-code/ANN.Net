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
            return (Quad)Math.Pow(Math.E, Math.Pow(-x, 2));
        }

        public Quad Derivative(Quad x)
        {
            return -2 * x * Activation(ref x);
        }
    }
}
