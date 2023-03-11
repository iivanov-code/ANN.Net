using System;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.ActivationFunctions;
using ANN.Net.LossFunctions;
using ANN.Net.Neurons;
using ANN.Net.Optimizers;

namespace ANN.Net.Utils
{
    public static class NetworkUtils
    {
        private static object idPadlock = new object();
        private static uint Id = 0;

        public static uint NewID()
        {
            uint id = Id;
            lock (idPadlock)
            {
                Id++;
            }

            return id;
        }

        public static Quad RandomWeight()
        {
            return NetworkUtils.GetRandomNumber(0, 1);
        }

        public static Quad XavierWeight(int neuronCount)
        {
            return 1 / NetworkUtils.GetRandomNumber(1, neuronCount) + NetworkUtils.GetRandomNumber(0.001f, 0.009f, 4);
        }

        public static Quad Normalize(Quad value, Quad minValue, Quad maxValue)
        {
            return (value - minValue) / (maxValue - minValue);
        }

        public static Quad Denormalize(Quad normalizedValue, Quad maxValue, Quad minValue)
        {
            return minValue + normalizedValue * maxValue - normalizedValue * minValue;
        }

        private static Random rand;

        internal static Random Rand
        {
            get
            {
                return rand ?? (rand = new Random(new Random().Next(int.MinValue, int.MaxValue)));
            }
        }

        internal static bool RandomBoolean()
        {
            return (Rand.Next() & 1) == 1;
        }

        internal static Quad GetRandomNumber(Quad fromInclusive, Quad toInclusive, int decimals = 2)
        {
            Quad value = Rand.GetRandomNumber(fromInclusive, toInclusive, decimals);
            value = (Quad)Math.Round(value, decimals);
            return value;
        }

        internal static Quad GetRandomNumber(this Random rand, Quad minimum, Quad maximum, int decimals = 6)
        {
            Quad value = (Quad)(rand.NextDouble() * (maximum - minimum) + minimum);
            return (Quad)Math.Round(value, decimals, MidpointRounding.AwayFromZero);
        }

        internal static ActivationTypes GetActivationType(this Neuron neuron)
        {
            switch (neuron.Function)
            {
                case ArcTanActivation _:
                    return ActivationTypes.ArcusTangens;

                case RELUActivation _:
                    return ActivationTypes.RELU;

                case SigmoidActivation _:
                    return ActivationTypes.Sigmoid;

                case TanHActivation _:
                    return ActivationTypes.HyperbolicTangens;

                default:
                    throw new ArgumentException("Invalid argument", nameof(neuron.Function));
            }
        }

        internal static NodeType GetNodeType(this INeuron neuron)
        {
            switch (neuron)
            {
                case InputNeuron _:
                    return NodeType.INPUT;

                case HiddenNeuron _:
                    return NodeType.HIDDEN;

                case OutputNeuron _:
                    return NodeType.OUTPUT;

                case BiasNeuron _:
                    return NodeType.BIAS;

                default:
                    throw new ArgumentException("Invalid neuron type", nameof(neuron));
            }
        }

        internal static IActivationFunction GetActivation(this Neuron neuron, ActivationTypes type)
        {
            return GetActivation(type);
        }

        internal static IActivationFunction GetActivation(ActivationTypes type)
        {
            switch (type)
            {
                case ActivationTypes.Sigmoid:
                    return SigmoidActivation.Instance;

                case ActivationTypes.HyperbolicTangens:
                    return TanHActivation.Instance;

                case ActivationTypes.ArcusTangens:
                    return ArcTanActivation.Instance;

                case ActivationTypes.RELU:
                    return RELUActivation.Instance;

                case ActivationTypes.LeakyRELU:
                    return LeakyRELUActivation.Instance;

                case ActivationTypes.BinaryRELU:
                    return BinaryStepActivation.Instance;

                default:
                    throw new ArgumentException("Invalid argument", nameof(type));
            }
        }

        internal static IOptimizer GetOptimizer(OptimizerTypes type)
        {
            switch (type)
            {
                case OptimizerTypes.Normal:
                    return new Normal();

                case OptimizerTypes.Adam:
                    return new Adam();

                case OptimizerTypes.AdaMax:
                    return new AdaMax();

                case OptimizerTypes.Nadam:
                    return new Nadam();

                case OptimizerTypes.RMSprop:
                    return new RMSprop();

                case OptimizerTypes.Momentum:
                    return new Momentum();

                case OptimizerTypes.Nesterov:
                    return new Nesterov();

                default:
                    throw new ArgumentException("Invalid optimizer type", nameof(type));
            }
        }

        internal static OptimizerTypes GetOptimizerType(this IOptimizer optimizer)
        {
            switch (optimizer)
            {
                case Normal _:
                    return OptimizerTypes.Normal;

                case Adam _:
                    return OptimizerTypes.Adam;

                case Nadam _:
                    return OptimizerTypes.Nadam;

                case AdaMax _:
                    return OptimizerTypes.AdaMax;

                case RMSprop _:
                    return OptimizerTypes.RMSprop;

                case Momentum _:
                    return OptimizerTypes.Momentum;

                case Nesterov _:
                    return OptimizerTypes.Nesterov;

                default:
                    throw new ArgumentException("Invalid optimizer instance type", nameof(optimizer));
            }
        }

        internal static ILossFunction GetLossFunction(LossFunctionTypes type, float alpha = 0)
        {
            switch (type)
            {
                case LossFunctionTypes.Absolute:
                    return AbsoluteLossFunction.Instance;

                case LossFunctionTypes.Squared:
                    return SquaredLossFunction.Instance;

                case LossFunctionTypes.Hinge:
                    return HingeLossFunction.Instance;

                case LossFunctionTypes.PseudoHuber:
                    return new PseudoHuberLossFunction(alpha);

                case LossFunctionTypes.Difference:
                    return DifferenceLossFunction.Instance;

                default:
                    throw new ArgumentException("Not implemented loss function", nameof(type));
            }
        }
    }
}
