using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    /// <summary>
    /// Parametric ReLU: x for x ≥ 0, alpha * x below. The slope is fixed (by default 0.25, the starting value PyTorch
    /// uses); ANN.Net does not learn it, so this is a leaky ReLU with a steeper negative side
    /// </summary>
    [Serializable]
    [ActivationType(Type = ActivationTypes.ParametricRELU)]
    internal class PReLUActivation : BaseActivation<PReLUActivation>, IActivationFunction
    {
        /// <summary>
        /// Slope of the negative side used for <see cref="ActivationTypes.ParametricRELU"/>
        /// </summary>
        public const float DefaultAlpha = 0.25f;

        private readonly Quad alpha;

        public PReLUActivation()
            : this(DefaultAlpha)
        {
        }

        public PReLUActivation(Quad alpha)
        {
            this.alpha = alpha;
        }

        public Quad MinValue => Quad.NegativeInfinity;

        public Quad MaxValue => Quad.PositiveInfinity;

        public Quad Activation(ref Quad x)
        {
            return x < 0 ? alpha * x : x;
        }

        /// <summary>
        /// Derivative at the value before activation: alpha below 0, 1 otherwise
        /// </summary>
        public Quad Derivative(Quad x)
        {
            return x < 0 ? alpha : 1;
        }
    }
}
