using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    [Serializable]
    [ActivationType(Type = ActivationTypes.TanHShrink)]
    internal class TanHShrinkActivation : BaseActivation<TanHShrinkActivation>, IActivationFunction
    {
        private static readonly Quad minValue = -1;
        private static readonly Quad maxValue = 1;

        public Quad MinValue => minValue;

        public Quad MaxValue => maxValue;

        public Quad Activation(ref Quad x)
        {
            x = x - (Quad)Math.Tanh(x);
            return x;
        }

        public Quad Derivative(Quad x)
        {
            return (Quad)(1 - Math.Pow(Math.Tanh(x), 2));
        }
    }
}
