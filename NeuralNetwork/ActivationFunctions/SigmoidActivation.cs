using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    [Serializable]
    [ActivationType(Type = ActivationTypes.Sigmoid)]
    internal class HardSigmoidActivation : BaseActivation<HardSigmoidActivation>, IActivationFunction
    {
        private static readonly Quad minValue = 0;
        private static readonly Quad maxValue = 1;

        public Quad MinValue
        {
            get
            {
                return minValue;
            }
        }

        public Quad MaxValue
        {
            get
            {
                return maxValue;
            }
        }

        public Quad Activation(ref Quad x)
        {
            x = (Quad)(1 / (1 + Math.Pow(Math.E, -x))); // standard sigmoid
            return x;
        }

        /// <summary>
        /// Derivative at the value before activation: s(x) * (1 - s(x))
        /// </summary>
        public Quad Derivative(Quad x)
        {
            double s = 1 / (1 + Math.Exp(-x));
            return (Quad)(s * (1 - s));
        }

        public static Quad Activate(ref Quad x)
        {
            return Instance.Activation(ref x);
        }
    }
}
