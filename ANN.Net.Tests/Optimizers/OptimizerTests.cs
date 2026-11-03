using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.IdentityGenerators;
using ANN.Net.Tests.Convolutional;
using ANN.Net.Utils;
using Xunit;

namespace ANN.Net.Tests.Optimizers
{
    /// <summary>
    /// Every optimizer against a double-precision reference of its published update rule, and training different
    /// kinds of networks with every optimizer. The networks start from seeded weights, so the results are reproducible
    /// </summary>
    public class OptimizerTests
    {
        private const double Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, Rho = 0.9;

        public static TheoryData<OptimizerTypes> AllOptimizers()
        {
            var data = new TheoryData<OptimizerTypes>();
            foreach (OptimizerTypes type in Enum.GetValues<OptimizerTypes>()) data.Add(type);
            return data;
        }

        /// <summary>
        /// Reference update rules (PyTorch conventions); the state lives in the closure
        /// </summary>
        private static Func<double, double, double> Reference(OptimizerTypes type, double lr)
        {
            double m = 0, v = 0, maxV = 0;
            int t = 0;

            return type switch
            {
                OptimizerTypes.Normal => (g, w) => w - (lr * g),
                OptimizerTypes.Momentum => (g, w) => { v = (Beta1 * v) + (lr * g); return w - v; },
                OptimizerTypes.Nesterov => (g, w) => { v = (Beta1 * v) + g; return w - (lr * (g + (Beta1 * v))); },
                OptimizerTypes.Adam => (g, w) =>
                {
                    t++;
                    m = (Beta1 * m) + ((1 - Beta1) * g);
                    v = (Beta2 * v) + ((1 - Beta2) * g * g);
                    double mHat = m / (1 - Math.Pow(Beta1, t)), vHat = v / (1 - Math.Pow(Beta2, t));
                    return w - (lr * mHat / (Math.Sqrt(vHat) + Epsilon));
                },
                OptimizerTypes.AdaMax => (g, w) =>
                {
                    t++;
                    m = (Beta1 * m) + ((1 - Beta1) * g);
                    v = Math.Max(Beta2 * v, Math.Abs(g));
                    return w - (lr / (1 - Math.Pow(Beta1, t)) * m / (v + Epsilon));
                },
                OptimizerTypes.Nadam => (g, w) =>
                {
                    t++;
                    m = (Beta1 * m) + ((1 - Beta1) * g);
                    v = (Beta2 * v) + ((1 - Beta2) * g * g);
                    double mHat = m / (1 - Math.Pow(Beta1, t)), vHat = v / (1 - Math.Pow(Beta2, t));
                    double nesterov = (Beta1 * mHat) + ((1 - Beta1) * g / (1 - Math.Pow(Beta1, t)));
                    return w - (lr * nesterov / (Math.Sqrt(vHat) + Epsilon));
                },
                OptimizerTypes.AmsGrad => (g, w) =>
                {
                    m = (Beta1 * m) + ((1 - Beta1) * g);
                    v = (Beta2 * v) + ((1 - Beta2) * g * g);
                    maxV = Math.Max(maxV, v);
                    return w - (lr * m / (Math.Sqrt(maxV) + Epsilon));
                },
                OptimizerTypes.RMSprop => (g, w) =>
                {
                    v = (Rho * v) + ((1 - Rho) * g * g);
                    return w - (lr * g / (Math.Sqrt(v) + Epsilon));
                },
                _ => throw new ArgumentException($"No reference for {type}", nameof(type))
            };
        }

        private static IOptimizer Create(OptimizerTypes type, float learningRate)
        {
            IOptimizer optimizer = NetworkUtils.GetOptimizer(type);
            if (optimizer is LearningRateOptimizer withRate) withRate.LearningRate = learningRate;
            return optimizer;
        }

        [Theory]
        [MemberData(nameof(AllOptimizers))]
        public void EveryOptimizer_CanBeCreatedAndRecognized(OptimizerTypes type)
        {
            IOptimizer optimizer = NetworkUtils.GetOptimizer(type);

            Assert.Equal(type, optimizer.GetOptimizerType());
            Assert.IsAssignableFrom<LearningRateOptimizer>(optimizer);
        }

        [Theory]
        [MemberData(nameof(AllOptimizers))]
        public void Update_MatchesThePublishedRule(OptimizerTypes type)
        {
            // starts with a zero gradient, which must not divide by zero
            double[] gradients = { 0, 0.5, -1.2, 2, 0, -0.3, 0.7, 1e-3, -4, 0.25, 0, 0, 1.5 };
            IOptimizer optimizer = Create(type, 0.1f);
            Func<double, double, double> reference = Reference(type, 0.1);
            double expected = 0.7;
            Quad weight = 0.7f;

            for (int i = 0; i < gradients.Length; i++)
            {
                expected = reference(gradients[i], expected);
                weight = optimizer.Calculate((Quad)gradients[i], weight);

                Assert.False(float.IsNaN(weight) || float.IsInfinity(weight), $"step {i}: {weight}");
                Assert.True(Math.Abs(expected - weight) <= 1e-4 * Math.Max(1, Math.Abs(expected)), $"step {i}: expected {expected}, got {(float)weight}");
            }
        }

        [Theory]
        [MemberData(nameof(AllOptimizers))]
        public void ZeroLearningRate_LeavesTheWeightUnchanged(OptimizerTypes type)
        {
            IOptimizer optimizer = Create(type, 0);
            Quad weight = 0.4f;

            foreach (float gradient in new[] { 0.3f, -2f, 0f, 5f })
            {
                weight = optimizer.Calculate(gradient, weight);
            }

            Assert.Equal(0.4f, (float)weight);
        }

        /// <summary>
        /// Learning rate that suits the optimizer: adaptive optimizers take steps of about the learning rate, plain
        /// gradient descent steps scale with the gradient. AdaMax divides by the largest recent gradient, which decays
        /// slowly, so it takes about twice the rate of Adam (as in the usual defaults, 0.002 against 0.001)
        /// </summary>
        private static float LearningRate(OptimizerTypes type) => type switch
        {
            OptimizerTypes.Normal => 0.05f,
            OptimizerTypes.Momentum or OptimizerTypes.Nesterov => 0.01f,
            OptimizerTypes.RMSprop => 0.003f,
            OptimizerTypes.AdaMax => 0.01f,
            _ => 0.005f
        };

        private static void SetLearningRate(IEnumerable<LearningRateOptimizer> optimizers, float learningRate)
        {
            foreach (LearningRateOptimizer optimizer in optimizers) optimizer.LearningRate = learningRate;
        }

        [Theory]
        [MemberData(nameof(AllOptimizers))]
        public void Optimizer_MinimizesAQuadratic(OptimizerTypes type)
        {
            IOptimizer optimizer = Create(type, type == OptimizerTypes.Normal ? 0.1f : 0.05f);
            Quad w = -2;

            // f(w) = (w - 3)², f'(w) = 2(w - 3)
            for (int i = 0; i < 2000; i++) w = optimizer.Calculate(2 * (w - 3), w);

            Assert.Equal(3f, (float)w, 1);
        }

        private static readonly Func<float, float, float> Target = (x0, x1) => (float)Math.Sin(2 * x0) + (0.5f * x1);

        private static (Quad[] Input, float Target)[] Samples(int count, int seed)
        {
            var random = new Random(seed);
            return Enumerable.Range(0, count).Select(_ =>
            {
                float x0 = (float)((random.NextDouble() * 2) - 1), x1 = (float)((random.NextDouble() * 2) - 1);
                return (new Quad[] { x0, x1 }, Target(x0, x1));
            }).ToArray();
        }

        private static double MeanSquaredError(INetwork network, (Quad[] Input, float Target)[] samples)
        {
            return samples.Average(s => Math.Pow((float)network.Propagate(s.Input)[0] - s.Target, 2));
        }

        private static void AssertLearnsRegression(INetwork network, string what)
        {
            var training = Samples(200, 1);
            var test = Samples(100, 2);
            double before = MeanSquaredError(network, test);

            for (int epoch = 0; epoch < 40; epoch++)
            {
                foreach (var (input, target) in training)
                {
                    network.Propagate(input);
                    network.Backpropagate(new Quad[] { target });
                }
            }

            double after = MeanSquaredError(network, test);
            Assert.False(double.IsNaN(after), $"{what}: the error became NaN");
            Assert.True(after < 0.03 && after < before / 4, $"{what}: test error went from {before:F4} to {after:F4}");
        }

        [Theory]
        [MemberData(nameof(AllOptimizers))]
        public void LayeredNetwork_LearnsARegression(OptimizerTypes type)
        {
            INetwork network = NetworkFactory.BuildFFN(new NetworkSettings
            {
                IdentityGenerator = new IncrementalGenerator(),
                LossType = LossFunctionTypes.Squared,
                Seed = 1,
                InputSettings = new InputLayerSettings { NeuronsCount = 2, ActivationType = ActivationTypes.Identity, HasBiasNeuron = true, OptimizerType = type },
                OutputSettings = new OutputLayerSettings { NeuronsCount = 1, ActivationType = ActivationTypes.Identity, OptimizerType = type }
            },
            new RecurrentLayerSettings
            {
                NeuronsCount = 10,
                LayersCount = 1,
                CellType = NeuronType.Normal,
                PrevLayerConnectionType = LayerConnectionType.FullyConnected,
                ActivationType = ActivationTypes.HyperbolicTangens,
                HasBiasNeuron = true,
                OptimizerType = type
            });
            SetLearningRate(((Network)network).Optimizers, LearningRate(type));

            AssertLearnsRegression(network, $"layered network with {type}");
        }

        [Theory]
        [MemberData(nameof(AllOptimizers))]
        public void GraphNetwork_LearnsARegression(OptimizerTypes type)
        {
            var random = new Random(3);
            var settings = new GraphNetworkSettings
            {
                Nodes = { new GraphNode(0, NodeType.INPUT), new GraphNode(1, NodeType.INPUT), new GraphNode(2, NodeType.BIAS), new GraphNode(100, NodeType.OUTPUT) },
                HiddenActivationType = ActivationTypes.HyperbolicTangens,
                OutputActivationType = ActivationTypes.Identity,
                OptimizerType = type
            };

            // two hidden columns of 5 nodes with skip connections from the inputs to the output
            uint id = 0;
            float Weight() => (float)((random.NextDouble() * 2) - 1);
            for (uint h = 10; h < 15; h++)
            {
                settings.Nodes.Add(new GraphNode(h, NodeType.HIDDEN));
                settings.Nodes.Add(new GraphNode(h + 10, NodeType.HIDDEN));
                foreach (uint source in new uint[] { 0, 1, 2 }) settings.Connections.Add(new GraphConnection(id++, source, h, Weight()));
                settings.Connections.Add(new GraphConnection(id++, h, h + 10, Weight()));
                settings.Connections.Add(new GraphConnection(id++, 2, h + 10, Weight()));
                settings.Connections.Add(new GraphConnection(id++, h + 10, 100, Weight() / 2));
            }

            settings.Connections.Add(new GraphConnection(id++, 0, 100, 0));
            settings.Connections.Add(new GraphConnection(id, 1, 100, 0));
            IGraphNetwork network = NetworkFactory.BuildGraph(settings);
            SetLearningRate(network.Optimizers, LearningRate(type));

            AssertLearnsRegression(network, $"graph network with {type}");
        }

        [Theory]
        [MemberData(nameof(AllOptimizers))]
        public void ConvolutionalNetwork_LearnsToClassifyBars(OptimizerTypes type)
        {
            IConvolutionalNetwork network = ConvolutionalNetworkTests.BarClassifier(type, LearningRate(type));
            var random = new Random(4);

            for (int i = 0; i < 1200; i++)
            {
                var (image, label) = ConvolutionalNetworkTests.Bar(random);
                network.Propagate(image);
                network.Backpropagate(label == 0 ? new Quad[] { 1, 0 } : new Quad[] { 0, 1 });
            }

            double accuracy = ConvolutionalNetworkTests.BarAccuracy(network, new Random(5));
            Assert.True(accuracy >= 0.95, $"accuracy {accuracy:P0} with {type}");
        }

        [Theory]
        [MemberData(nameof(AllOptimizers))]
        public void LayeredConvolution_LearnsATranslationInvariantPattern(OptimizerTypes type)
        {
            Network network = Networks.LocalConnectionTests.Build(10, LayerConnectionType.Convolutional, kernel: 2, stride: 1, padding: 0, filters: 4,
                                                                  inputBias: true, hiddenActivation: ActivationTypes.RELU, optimizer: type);

            // shared kernel weights sum the gradients of 9 positions, so plain gradient descent needs a smaller rate
            SetLearningRate(network.Optimizers, type == OptimizerTypes.Normal ? 0.005f : LearningRate(type) / 2);

            var random = new Random(6);
            for (int i = 0; i < 6000; i++)
            {
                var (signal, label) = Networks.LocalConnectionTests.Edge(random);
                network.Propagate(signal);
                network.Backpropagate(new Quad[] { label });
            }

            var test = new Random(7);
            int correct = Enumerable.Range(0, 200).Count(_ =>
            {
                var (signal, label) = Networks.LocalConnectionTests.Edge(test);
                return Math.Abs((float)network.Propagate(signal)[0] - label) < 0.5f;
            });

            Assert.True(correct >= 180, $"{type}: {correct} of 200 correct");
        }

        [Theory]
        [MemberData(nameof(AllOptimizers))]
        public void Transformer_LearnsNextTokenPrediction(OptimizerTypes type)
        {
            ISequenceNetwork network = NetworkFactory.BuildTransformer(new TransformerSettings
            {
                InputDimension = 4,
                ModelDimension = 8,
                HeadsCount = 2,
                FeedForwardDimension = 16,
                LayersCount = 2,
                OutputDimension = 4,
                MaxSequenceLength = 6,
                UseCausalMask = true,
                OutputApplySoftMax = true,
                OptimizerType = type,
                InitializationType = WeightInitType.Xavier,
                Seed = 1
            });
            SetLearningRate(network.Optimizers, type == OptimizerTypes.Normal ? 0.05f : LearningRate(type));

            Quad[][] OneHot(params int[] tokens) => tokens.Select(t => Enumerable.Range(0, 4).Select(i => (Quad)(i == t ? 1 : 0)).ToArray()).ToArray();
            Quad[][] inputs = OneHot(0, 1, 2, 3, 0), targets = OneHot(1, 2, 3, 0, 1);

            network.Propagate(inputs);
            float initial = network.Backpropagate(targets), final = initial;
            for (int epoch = 0; epoch < 300; epoch++)
            {
                network.Propagate(inputs);
                final = network.Backpropagate(targets);
            }

            Assert.True(final < initial / 4, $"{type}: error went from {initial} to {final}");
            Quad[][] outputs = network.Propagate(inputs);
            for (int t = 0; t < inputs.Length; t++)
            {
                Assert.Equal(Array.IndexOf(targets[t], (Quad)1), Array.IndexOf(outputs[t], outputs[t].Max()));
            }
        }
    }
}
