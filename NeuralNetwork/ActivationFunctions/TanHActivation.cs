using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    [Serializable]
    [ActivationType(Type = ActivationTypes.HyperbolicTangens)]
    internal class TanHActivation : BaseActivation<TanHActivation>, IActivationFunction
    {
        private static readonly Quad minValue = -1;
        private static readonly Quad maxValue = 1;

        public Quad MinValue => minValue;

        public Quad MaxValue => maxValue;

        public Quad Activation(ref Quad x)
        {
            x = (Quad)Math.Tanh(x);
            return x;
        }

        /// <summary>
        /// Derivative at the value before activation: 1 - tanh(x)^2
        /// </summary>
        public Quad Derivative(Quad x)
        {
            double tanh = Math.Tanh(x);
            return (Quad)(1 - tanh * tanh);
        }

        public static Quad Activate(ref Quad x)
        {
            return Instance.Activation(ref x);
        }
    }
}
