using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.HelperClasses;
using System;

namespace ANN.Net.ActivationFunctions
{
    [ActivationType(Type = ActivationTypes.Sigmoid)]
    internal class SigmoidActivation : BaseActivation<SigmoidActivation>, IActivationFunction
    {
        private static readonly float minValue = 0;
        private static readonly float maxValue = 1;

        public float MinValue
        {
            get
            {
                return minValue;
            }
        }

        public float MaxValue
        {
            get
            {
                return maxValue;
            }
        }

        public float Activation(float x)
        {
            return (float)(1 / (1 + Math.Pow(Math.E, -x))); // standard sigmoid
        }

        public float Prime(float deltaX)
        {
            return deltaX * (1 - deltaX);
        }

        public static float Activate(float x)
        {
            return Instance.Activation(x);
        }
    }
}
