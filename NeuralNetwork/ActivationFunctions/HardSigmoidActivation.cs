using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    /// <summary>
    /// Piecewise linear sigmoid: max(0, min(1, x / 6 + 1 / 2)), as in PyTorch, so that hard swish is x times it
    /// </summary>
    [Serializable]
    [ActivationType(Type = ActivationTypes.HardSigmoid)]
    internal class HardSigmoidActivation : BaseActivation<HardSigmoidActivation>, IActivationFunction
    {
        /// <summary>
        /// Slope of the linear part
        /// </summary>
        public const float Slope = 1f / 6;

        /// <summary>
        /// Value at 0
        /// </summary>
        public const float Offset = 0.5f;

        public Quad MinValue => 0;

        public Quad MaxValue => 1;

        public Quad Activation(ref Quad x)
        {
            return (Quad)Math.Clamp(Slope * (double)x + Offset, 0, 1);
        }

        /// <summary>
        /// Derivative at the value before activation: the slope between -3 and 3, 0 outside
        /// </summary>
        public Quad Derivative(Quad x)
        {
            double linear = Slope * (double)x + Offset;
            return linear > 0 && linear < 1 ? Slope : 0;
        }
    }
}
