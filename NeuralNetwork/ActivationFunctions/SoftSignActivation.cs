using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    [Serializable]
    [ActivationType(Type = ActivationTypes.SoftSign)]
    internal class SoftSignActivation : BaseActivation<SoftSignActivation>, IActivationFunction
    {
        public Quad MinValue => -1;

        public Quad MaxValue => 1;

        public Quad Activation(ref Quad x)
        {
            return x / (1 + Math.Abs(x));
        }

        public Quad Derivative(Quad x)
        {
            return (Quad)(1 / Math.Pow(1 + Math.Abs(x), 2));
        }
    }
}
