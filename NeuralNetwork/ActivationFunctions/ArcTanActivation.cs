﻿using NeuralNetwork.Enums;
using NeuralNetwork.HelperClasses;
using NeuralNetwork.Interfaces;
using System;

namespace NeuralNetwork.ActivationFunctions
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
