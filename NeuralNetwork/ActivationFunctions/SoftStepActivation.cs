using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.HelperClasses;
using System;

namespace ANN.Net.ActivationFunctions
{
    [Serializable]
    [ActivationType(Type = ActivationTypes.SoftStep)]
    internal class SoftStepActivation : BaseActivation<SoftStepActivation>, IActivationFunction
    {
        private static readonly float minValue = 0;
        private static readonly float maxValue = 1;

        public float MinValue => minValue;

        public float MaxValue => maxValue;

        public float Activation(float x)
        {
            return (float)(1 / (1 + Math.Pow(Math.E, -1 * x)));
        }

        public float Prime(float x)
        {
            return x * (1 - x);
        }
    }
}
