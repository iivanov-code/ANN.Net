using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    [Serializable]
    [ActivationType(Type = ActivationTypes.Sinusoid)]
    internal class SinusoidActivation : BaseActivation<SinusoidActivation>, IActivationFunction
    {
        public Quad MinValue => -1;

        public Quad MaxValue => 1;

        public Quad Activation(ref Quad x)
        {
            return (Quad)Math.Sin(x);
        }

        public Quad Derivative(Quad x)
        {
            return (Quad)Math.Cos(x);
        }
    }
}
