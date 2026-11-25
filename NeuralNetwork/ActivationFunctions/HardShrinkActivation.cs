using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    /// <summary>
    /// Hard shrinkage: x outside [-lambda, lambda], 0 inside (lambda 0.5 by default, as in PyTorch)
    /// </summary>
    [Serializable]
    [ActivationType(Type = ActivationTypes.HardShrink)]
    internal class HardShrinkActivation : BaseActivation<HardShrinkActivation>, IActivationFunction
    {
        /// <summary>
        /// Threshold used for <see cref="ActivationTypes.HardShrink"/>
        /// </summary>
        public const float DefaultLambda = 0.5f;

        private readonly Quad lambda;

        public HardShrinkActivation()
            : this(DefaultLambda)
        {
        }

        public HardShrinkActivation(Quad lambda)
        {
            this.lambda = lambda;
        }

        public Quad MinValue => Quad.NegativeInfinity;

        public Quad MaxValue => Quad.PositiveInfinity;

        public Quad Activation(ref Quad x)
        {
            return x > lambda || x < -lambda ? x : 0;
        }

        /// <summary>
        /// Derivative at the value before activation: 1 outside [-lambda, lambda], 0 inside
        /// </summary>
        public Quad Derivative(Quad x)
        {
            return x > lambda || x < -lambda ? 1 : 0;
        }
    }
}
