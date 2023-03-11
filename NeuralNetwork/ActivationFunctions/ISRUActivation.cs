using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    /// <summary>
    /// Inverse square root unit
    /// </summary>
    [ActivationType(Type = ActivationTypes.ISRU)]
    public class ISRUActivation : IActivationFunction
    {
        private Quad alpha;

        public ISRUActivation(Quad alpha)
        {
            this.alpha = alpha;
            this.MinValue = -(Quad)(1 / Math.Sqrt(alpha));
            this.MaxValue = (Quad)(1 / Math.Sqrt(alpha));
        }

        public Quad MinValue { get; }

        public Quad MaxValue { get; }

        public Quad Activation(ref Quad x)
        {
            return (Quad)(x / Math.Sqrt(1 + alpha * Math.Pow(x, 2)));
        }

        public Quad Derivative(Quad x)
        {
            return (Quad)Math.Pow(1 / Math.Sqrt(1 + alpha * Math.Pow(x, 2)), 3);
        }
    }
}
