using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    /// <summary>
    /// x - tanh(x)
    /// </summary>
    [Serializable]
    [ActivationType(Type = ActivationTypes.TanHShrink)]
    internal class TanHShrinkActivation : BaseActivation<TanHShrinkActivation>, IActivationFunction
    {
        public Quad MinValue => Quad.NegativeInfinity;

        public Quad MaxValue => Quad.PositiveInfinity;

        public Quad Activation(ref Quad x)
        {
            double value = x;
            return (Quad)(value - Math.Tanh(value));
        }

        /// <summary>
        /// Derivative at the value before activation: 1 - (1 - tanh(x)^2) = tanh(x)^2
        /// </summary>
        public Quad Derivative(Quad x)
        {
            double tanh = Math.Tanh(x);
            return (Quad)(tanh * tanh);
        }
    }
}
