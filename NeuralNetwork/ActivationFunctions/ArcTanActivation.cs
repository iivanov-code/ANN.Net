using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.HelperClasses;
using System;

namespace ANN.Net.ActivationFunctions
{
    [ActivationType(Type = ActivationTypes.ArcusTangens)]
    internal class ArcTanActivation : BaseActivation<ArcTanActivation>, IActivationFunction
    {
        private static readonly float minValue = -1;
        private static readonly float maxValue = 1;

        public float MinValue => minValue;

        public float MaxValue => maxValue;

        public float Activation(float x)
        {
            return (float)Math.Atan(x);
        }

        public float Prime(float x)
        {
            return (float)(1 / (Math.Pow(x, 2) + 1));
        }
    }
}
