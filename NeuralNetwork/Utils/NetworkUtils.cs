using System;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.ActivationFunctions;
using ANN.Net.Neurons;

namespace ANN.Net.Utils
{
    internal static class NetworkUtils
    {
        public static float Normalize(float value, float minValue, float maxValue)
        {
            return (value - minValue) / (maxValue - minValue);
        }

        public static float UnNormalize(float normalizedValue, float maxValue, float minValue)
        {
            return minValue + normalizedValue * maxValue - normalizedValue * minValue;
        }

        private static Random rand;

        public static Random Rand
        {
            get
            {
                return rand ?? (rand = new Random(new Random().Next(int.MinValue, int.MaxValue)));
            }
        }

        public static bool RandomBoolean()
        {
            return (Rand.Next() & 1) == 1;
        }

        public static float GetRandomNumber(float fromInclusive, float toInclusive, int decimals = 2)
        {
            float value = Rand.GetRandomNumber(fromInclusive, toInclusive, decimals);
            value = (float)Math.Round(value, decimals);
            return value;
        }

        public static float GetRandomNumber(this Random rand, float minimum, float maximum, int decimals = 6)
        {
            float value = (float)(rand.NextDouble() * (maximum - minimum) + minimum);
            return (float)Math.Round(value, decimals, MidpointRounding.AwayFromZero);
        }

        public static ActivationTypes GetActivationType(this Neuron neuron, IActivationFunction activation)
        {
            switch (activation)
            {
                case ArcTanActivation _:
                    return ActivationTypes.ArcusTangens;
                case RELUActivation _:
                    return ActivationTypes.RELU;
                case SigmoidActivation _:
                    return ActivationTypes.Sigmoid;
                case SoftPlusActivation _:
                    return ActivationTypes.SoftPlus;
                case SoftStepActivation _:
                    return ActivationTypes.SoftStep;
                case TanHActivation _:
                    return ActivationTypes.HyperbolicTangens;
                default:
                    throw new ArgumentException("Invalid argument", nameof(activation));
            }
        }

        public static IActivationFunction GetActivation(this Neuron neuron, ActivationTypes type)
        {
            return GetActivation(type);
        }

        public static IActivationFunction GetActivation(ActivationTypes type)
        {
            switch (type)
            {
                case ActivationTypes.Sigmoid:
                    return SigmoidActivation.Instance;
                case ActivationTypes.SoftPlus:
                    return SoftPlusActivation.Instance;
                case ActivationTypes.SoftStep:
                    return SoftStepActivation.Instance;
                case ActivationTypes.HyperbolicTangens:
                    return TanHActivation.Instance;
                case ActivationTypes.ArcusTangens:
                    return ArcTanActivation.Instance;
                case ActivationTypes.RELU:
                    return RELUActivation.Instance;
                default:
                    throw new ArgumentException("Invalid argument", nameof(type));
            }
        }
    }
}