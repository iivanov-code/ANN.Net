using System;
using ANN.Net.Abstractions.Settings;
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
        private static int lastId = -1;

        /// <summary>
        /// A new id, unique also when networks are built on several threads at once
        /// </summary>
        public static uint NewID()
        {
            return unchecked((uint)System.Threading.Interlocked.Increment(ref lastId));
        }

        public static Quad RandomWeight()
        {
            return NetworkUtils.GetRandomNumber(0, 1);
        }

        public static Quad XavierWeight(int neuronCount)
        {
            return 1 / NetworkUtils.GetRandomNumber(1, neuronCount) + NetworkUtils.GetRandomNumber(0.001f, 0.009f, 4);
        }

        /// <summary>
        /// Glorot (Xavier) uniform initialization: U(-limit, limit) with limit = sqrt(6 / (fanIn + fanOut))
        /// </summary>
        public static Quad GlorotWeight(int fanIn, int fanOut)
        {
            double limit = System.Math.Sqrt(6.0 / (fanIn + fanOut));
            return (Quad)(Rand.NextDouble() * 2 * limit - limit);
        }

        public static Quad Normalize(Quad value, Quad minValue, Quad maxValue)
        {
            return (value - minValue) / (maxValue - minValue);
        }

        public static Quad Denormalize(Quad normalizedValue, Quad maxValue, Quad minValue)
        {
            return minValue + normalizedValue * maxValue - normalizedValue * minValue;
        }

        /// <summary>
        /// Shared random source for weight initialization and sampling.
        /// <see cref="Random.Shared"/> is safe to use from several threads at once, which a
        /// hand-made <see cref="Random"/> is not: concurrent calls can leave it returning zeros
        /// </summary>
        internal static Random Rand => scopedRandom.Value ?? Random.Shared;

        private static readonly System.Threading.AsyncLocal<Random> scopedRandom = new System.Threading.AsyncLocal<Random>();

        /// <summary>
        /// Draws every random number of the current flow from a source seeded with <paramref name="seed"/> until the
        /// returned scope is disposed, so a network built inside the scope gets reproducible initial weights. Does
        /// nothing for a null seed. The scope follows the async flow, so networks built in parallel do not interfere
        /// </summary>
        internal static IDisposable SeedRandom(int? seed)
        {
            return seed.HasValue ? new RandomScope(new Random(seed.Value)) : RandomScope.None;
        }

        private sealed class RandomScope : IDisposable
        {
            public static readonly RandomScope None = new RandomScope();

            private readonly Random previous;
            private readonly bool active;

            private RandomScope()
            {
            }

            public RandomScope(Random random)
            {
                previous = scopedRandom.Value;
                scopedRandom.Value = random;
                active = true;
            }

            public void Dispose()
            {
                if (active) scopedRandom.Value = previous;
            }
        }

        internal static bool RandomBoolean()
        {
            return (Rand.Next() & 1) == 1;
        }

        internal static Quad GetRandomNumber(Quad fromInclusive, Quad toInclusive, int decimals = 2)
        {
            Quad value = Rand.GetRandomNumber(fromInclusive, toInclusive, decimals);
            value = (Quad)System.Math.Round(value, decimals);
            return value;
        }

        internal static Quad GetRandomNumber(this Random rand, Quad minimum, Quad maximum, int decimals = 6)
        {
            Quad value = (Quad)(rand.NextDouble() * (maximum - minimum) + minimum);
            return (Quad)System.Math.Round(value, decimals, MidpointRounding.AwayFromZero);
        }

        internal static ActivationTypes GetActivationType(this Neuron neuron)
        {
            switch (neuron.Function)
            {
                case ArcTanActivation _:
                    return ActivationTypes.ArcusTangens;

                case RELUActivation _:
                    return ActivationTypes.RELU;

                case HardSigmoidActivation _:
                    return ActivationTypes.Sigmoid;

                case TanHActivation _:
                    return ActivationTypes.HyperbolicTangens;

                case IdentityActivation _:
                    return ActivationTypes.Identity;

                case LeakyRELUActivation _:
                    return ActivationTypes.LeakyRELU;

                case BinaryStepActivation _:
                    return ActivationTypes.BinaryRELU;

                case GaussianActivation _:
                    return ActivationTypes.Gaussian;

                case SinusoidActivation _:
                    return ActivationTypes.Sinusoid;

                case SoftSignActivation _:
                    return ActivationTypes.SoftSign;

                case SQNLActivation _:
                    return ActivationTypes.SQNL;

                case ISRUActivation _:
                    return ActivationTypes.ISRU;

                case GELUActivation _:
                    return ActivationTypes.GELU;

                case SwishActivation _:
                    return ActivationTypes.SWISH;

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
                    return HardSigmoidActivation.Instance;

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

                case ActivationTypes.Identity:
                    return IdentityActivation.Instance;

                case ActivationTypes.Gaussian:
                    return GaussianActivation.Instance;

                case ActivationTypes.Sinusoid:
                    return SinusoidActivation.Instance;

                case ActivationTypes.SoftSign:
                    return SoftSignActivation.Instance;

                case ActivationTypes.SQNL:
                    return SQNLActivation.Instance;

                case ActivationTypes.ISRU:
                    return ISRUActivation.Instance;

                case ActivationTypes.GELU:
                    return GELUActivation.Instance;

                case ActivationTypes.SWISH:
                    return SwishActivation.Instance;

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

                case OptimizerTypes.AmsGrad:
                    return new AmsGrad();

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

                case AmsGrad _:
                    return OptimizerTypes.AmsGrad;

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

        /// <summary>
        /// The loss of <paramref name="type"/>; <paramref name="options"/> null uses the default parameters
        /// </summary>
        internal static ILossFunction GetLossFunction(LossFunctionTypes type, LossOptions options = null)
        {
            options ??= new LossOptions();
            options.Validate();

            return type switch
            {
                LossFunctionTypes.Difference => new DifferenceLossFunction(),
                LossFunctionTypes.Absolute => new MeanAbsoluteLossFunction(),
                LossFunctionTypes.Squared => new SquaredLossFunction(),
                LossFunctionTypes.Hinge => new HingeLossFunction(),
                LossFunctionTypes.PseudoHuber => new PseudoHuberLossFunction(options.Delta),
                LossFunctionTypes.MeanSquared => new MeanSquareLossFunction(),
                LossFunctionTypes.Huber => new HuberLossFunction(options.Delta),
                LossFunctionTypes.SmoothL1 => new SmoothL1LossFunction(options.Delta),
                LossFunctionTypes.LogCosh => new LogCoshLossFunction(),
                LossFunctionTypes.MeanSquaredLogarithmic => new MeanSquaredLogarithmicLossFunction(),
                LossFunctionTypes.MeanAbsolutePercentage => new MeanAbsolutePercentageLossFunction(),
                LossFunctionTypes.Poisson => new PoissonLossFunction(),
                LossFunctionTypes.CrossEntropy => new CrossEntropyLossFunction(),
                LossFunctionTypes.BinaryCrossEntropy => new BinaryCrossEntropyLossFunction(),
                LossFunctionTypes.KullbackLeibler => new KullbackLeiblerLossFunction(),
                LossFunctionTypes.Focal => new FocalLossFunction(options.FocalGamma, options.FocalAlpha),
                LossFunctionTypes.SquaredHinge => new SquaredHingeLossFunction(),
                LossFunctionTypes.SoftMargin => new SoftMarginLossFunction(),
                LossFunctionTypes.CosineDistance => new CosineDistanceLossFunction(),
                _ => throw new ArgumentException("Not implemented loss function", nameof(type)),
            };
        }
    }
}
