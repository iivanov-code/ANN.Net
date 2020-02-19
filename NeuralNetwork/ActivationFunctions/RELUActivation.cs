using NeuralNetwork.Enums;
using NeuralNetwork.HelperClasses;
using NeuralNetwork.Interfaces;
using System;

namespace NeuralNetwork.ActivationFunctions
{
    [ActivationType(Type = ActivationTypes.RELU)]
    internal class RELUActivation : BaseActivation<RELUActivation>, IActivationFunction
    {
        private static readonly float minValue = 0;
        private static readonly float maxValue = 1;

        public float MinValue => minValue;

        public float MaxValue => maxValue;

        public float Activation(float x)
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

        public float Prime(float x)
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
    }
}
