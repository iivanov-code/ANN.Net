using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    [Serializable]
    [ActivationType(Type = ActivationTypes.Sigmoid)]
    internal class SigmoidActivation : BaseActivation<SigmoidActivation>, IActivationFunction
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

        public Quad Derivative(Quad deltaX)
        {
            return deltaX * (1 - deltaX);
        }

        public static Quad Activate(ref Quad x)
        {
            return Instance.Activation(ref x);
        }
    }
}
