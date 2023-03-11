using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    [Serializable]
    [ActivationType(Type = ActivationTypes.RELU)]
    internal class RELUActivation : BaseActivation<RELUActivation>, IActivationFunction
    {
        private static readonly Quad minValue = 0;
        private static readonly Quad maxValue = Quad.PositiveInfinity;

        public Quad MinValue => minValue;

        public Quad MaxValue => maxValue;

        public Quad Activation(ref Quad x)
        {
            if (x < 0)
            {
                return 0;
            }
            else
            {
                return x;
            }
        }

        public Quad Derivative(Quad x)
        {
            if (x < 0)
            {
                return 0;
            }
            else
            {
                return 1;
            }
        }

        public static Quad Activate(ref Quad x)
        {
            return Instance.Activation(ref x);
        }
    }
}
