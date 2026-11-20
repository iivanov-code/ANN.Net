using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    /// <summary>
    /// Soft shrinkage: x - lambda above lambda, x + lambda below -lambda, 0 between (lambda 0.5 by default, as in PyTorch)
    /// </summary>
    [Serializable]
    [ActivationType(Type = ActivationTypes.SoftShrink)]
    internal class SoftShrinkActivation : BaseActivation<SoftShrinkActivation>, IActivationFunction
    {
        /// <summary>
        /// Threshold used for <see cref="ActivationTypes.SoftShrink"/>
        /// </summary>
        public const float DefaultLambda = 0.5f;

        private readonly Quad lambda;

        public SoftShrinkActivation()
            : this(DefaultLambda)
        {
        }

        public SoftShrinkActivation(Quad lambda)
        {
            this.lambda = lambda;
        }

        public Quad MinValue => Quad.NegativeInfinity;

        public Quad MaxValue => Quad.PositiveInfinity;

        public Quad Activation(ref Quad x)
        {
            if (x > lambda)
            {
                return x - lambda;
            }
            else if (x < -lambda)
            {
                return x + lambda;
            }
            else
            {
                return 0;
            }
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
