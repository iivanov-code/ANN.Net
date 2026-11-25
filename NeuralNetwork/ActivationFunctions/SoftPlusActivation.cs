using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    /// <summary>
    /// SoftPlus: ln(1 + e^x), a smooth ReLU
    /// </summary>
    [Serializable]
    [ActivationType(Type = ActivationTypes.SoftPlus)]
    internal class SoftPlusActivation : BaseActivation<SoftPlusActivation>, IActivationFunction
    {
        public Quad MinValue => 0;

        public Quad MaxValue => Quad.PositiveInfinity;

        public Quad Activation(ref Quad x)
        {
            double value = x;

            // max(x, 0) + ln(1 + e^-|x|) does not overflow for large x
            return (Quad)(Math.Max(value, 0) + Math.Log(1 + Math.Exp(-Math.Abs(value))));
        }

        /// <summary>
        /// Derivative at the value before activation: the sigmoid 1 / (1 + e^-x)
        /// </summary>
        public Quad Derivative(Quad x)
        {
            return (Quad)(1 / (1 + Math.Exp(-(double)x)));
        }
    }
}
