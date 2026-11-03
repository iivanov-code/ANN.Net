using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Convolutional;
using ANN.Net.Tests.Transformers;
using Xunit;

namespace ANN.Net.Tests.Convolutional
{
    public class ConvolutionalNetworkTests
    {
        private static Quad[] RandomValues(Random random, int count, double scale = 1)
        {
            return Enumerable.Range(0, count).Select(_ => (Quad)(((random.NextDouble() * 2) - 1) * scale)).ToArray();
        }

        /// <summary>
        /// Values at least 0.05 apart in random order, so a maximum cannot change within the finite difference step
        /// </summary>
        private static Quad[] DistinctValues(Random random, int count)
        {
            return Enumerable.Range(0, count).Select(i => (Quad)((i * 0.05f) - 1)).OrderBy(_ => random.Next()).ToArray();
        }

        private static double Loss(Quad[] outputs, Quad[] upstream) => outputs.Zip(upstream, (o, u) => (double)(float)o * (float)u).Sum();

        /// <summary>
        /// Input gradient against central differences of L = Σ outputs * upstream, with learning rate 0 so nothing changes
        /// </summary>
        internal static void AssertInputGradient(IConvolutionalLayer layer, Random random, bool distinct = false)
        {
            Quad[] input = distinct ? DistinctValues(random, layer.InputShape.Size) : RandomValues(random, layer.InputShape.Size);
            Quad[] upstream = RandomValues(random, layer.OutputShape.Size);

            layer.Propagate(input);
            Quad[] analytic = layer.Backpropagate(upstream);

            for (int i = 0; i < input.Length; i++)
            {
                Quad original = input[i];
                input[i] = original + (Quad)GradientChecker.EPSILON;
                double plus = Loss(layer.Propagate(input), upstream);
                input[i] = original - (Quad)GradientChecker.EPSILON;
                double minus = Loss(layer.Propagate(input), upstream);
                input[i] = original;
                GradientChecker.AssertClose((plus - minus) / (2 * GradientChecker.EPSILON), (float)analytic[i], $"input gradient [{i}]");
            }
        }

        /// <summary>
        /// Parameter gradients: one plain gradient step with learning rate 1 changes every value by exactly its gradient
        /// </summary>
        internal static void AssertParameterGradients(IConvolutionalLayer layer, Parameters parameters, ICollection<LearningRateOptimizer> optimizers, Random random)
        {
            Quad[] input = RandomValues(random, layer.InputShape.Size);
            Quad[] upstream = RandomValues(random, layer.OutputShape.Size);

            var numeric = new double[parameters.Values.Length];
            for (int i = 0; i < numeric.Length; i++)
            {
                Quad original = parameters.Values[i];
                parameters.Values[i] = original + (Quad)GradientChecker.EPSILON;
                double plus = Loss(layer.Propagate(input), upstream);
                parameters.Values[i] = original - (Quad)GradientChecker.EPSILON;
                double minus = Loss(layer.Propagate(input), upstream);
                parameters.Values[i] = original;
                numeric[i] = (plus - minus) / (2 * GradientChecker.EPSILON);
            }

            Quad[] before = (Quad[])parameters.Values.Clone();
            foreach (LearningRateOptimizer optimizer in optimizers) optimizer.LearningRate = 1;
            layer.Propagate(input);
            layer.Backpropagate(upstream);

            for (int i = 0; i < numeric.Length; i++)
            {
                GradientChecker.AssertClose(numeric[i], (float)before[i] - (float)parameters.Values[i], $"parameter gradient [{i}]");
            }
        }

        public static TheoryData<int, int, int, int> ConvolutionCases() => new TheoryData<int, int, int, int>
        {
            // kernel, stride, padding, channels
            { 3, 1, 0, 1 },
            { 3, 1, 1, 2 },
            { 3, 2, 1, 3 },
            { 2, 2, 0, 2 },
            { 1, 1, 0, 2 }
        };

        [Theory]
        [MemberData(nameof(ConvolutionCases))]
        public void Convolution_GradientsMatchFiniteDifferences(int kernel, int stride, int padding, int channels)
        {
            var random = new Random(kernel * 100 + stride * 10 + padding);
            var optimizers = new List<LearningRateOptimizer>();
            var layer = new ConvolutionLayer(new Shape(channels, 6, 5), 3, kernel, kernel, stride, padding, ActivationTypes.HyperbolicTangens,
                                             OptimizerTypes.Normal, WeightInitType.Xavier, optimizers);

            AssertInputGradient(layer, random);
            AssertParameterGradients(layer, layer.Weights, optimizers, random);
            AssertParameterGradients(layer, layer.Biases, optimizers, random);
        }

        [Theory]
        [InlineData(PoolingType.Max, 2, 2)]
        [InlineData(PoolingType.Max, 3, 1)]
        [InlineData(PoolingType.Average, 2, 2)]
        [InlineData(PoolingType.Average, 3, 2)]
        public void Pooling_GradientsMatchFiniteDifferences(PoolingType type, int size, int stride)
        {
            // max pooling has a kink where two values of a window are equal, so its inputs are kept apart
            AssertInputGradient(new PoolingLayer(new Shape(2, 7, 6), type, size, stride), new Random(size * 10 + stride), distinct: type == PoolingType.Max);
        }

        [Fact]
        public void Dense_GradientsMatchFiniteDifferences()
        {
            var random = new Random(5);
            var optimizers = new List<LearningRateOptimizer>();
            var layer = new DenseLayer(new Shape(2, 3, 2), 4, ActivationTypes.Sigmoid, OptimizerTypes.Normal, WeightInitType.Xavier, optimizers);

            AssertInputGradient(layer, random);
            AssertParameterGradients(layer, layer.Weights, optimizers, random);
            AssertParameterGradients(layer, layer.Biases, optimizers, random);
        }

        [Fact]
        public void Convolution_ComputesTheKnownResult()
        {
            var layer = new ConvolutionLayer(new Shape(1, 3, 3), 1, 2, 2, 1, 0, ActivationTypes.Identity, OptimizerTypes.Normal, WeightInitType.Xavier, new List<LearningRateOptimizer>());
            Quad[] kernel = { 1, 0, 0, -1 };   // x[r, c] - x[r + 1, c + 1]
            kernel.CopyTo(layer.Weights.Values, 0);
            layer.Biases.Values[0] = 0.5f;

            Quad[] output = layer.Propagate(new Quad[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 });

            Assert.Equal(new[] { 1 - 5 + 0.5f, 2 - 6 + 0.5f, 4 - 8 + 0.5f, 5 - 9 + 0.5f }, output.Select(x => (float)x));
            Assert.Equal(new Shape(1, 2, 2), layer.OutputShape);
        }

        private static ConvolutionalNetworkSettings SmallNetwork(bool softMax) => new ConvolutionalNetworkSettings
        {
            InputChannels = 2,
            InputHeight = 6,
            InputWidth = 6,
            Layers =
            {
                new ConvolutionLayerSettings { Filters = 3, KernelSize = 3, Padding = 1, ActivationType = ActivationTypes.HyperbolicTangens },
                new PoolingLayerSettings { Type = PoolingType.Average, Size = 2 },
                new ConvolutionLayerSettings { Filters = 2, KernelSize = 2, Stride = 1, ActivationType = ActivationTypes.Sigmoid },
                new DenseLayerSettings { NeuronsCount = 5, ActivationType = ActivationTypes.HyperbolicTangens }
            },
            OutputDimension = 3,
            OutputApplySoftMax = softMax,
            OutputActivationType = ActivationTypes.Sigmoid
        };

        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public void Network_InputGradientsMatchFiniteDifferences(bool softMax)
        {
            var random = new Random(softMax ? 11 : 12);
            IConvolutionalNetwork network = NetworkFactory.BuildConvolutional(SmallNetwork(softMax));
            Quad[] input = RandomValues(random, 2 * 6 * 6);
            Quad[] targets = softMax ? new Quad[] { 0, 1, 0 } : RandomValues(random, 3);

            double LossOf(Quad[] values)
            {
                Quad[] outputs = network.Propagate(values);
                return softMax
                    ? -Math.Log((float)outputs[1])
                    : outputs.Zip(targets, (o, t) => 0.5 * Math.Pow((float)o - (float)t, 2)).Sum();
            }

            network.Propagate(input);
            Assert.Equal(LossOf(input), (float)network.Backpropagate(targets), 1e-4);
            Quad[] analytic = network.InputGradients;

            for (int i = 0; i < input.Length; i++)
            {
                Quad original = input[i];
                input[i] = original + (Quad)GradientChecker.EPSILON;
                double plus = LossOf(input);
                input[i] = original - (Quad)GradientChecker.EPSILON;
                double minus = LossOf(input);
                input[i] = original;
                GradientChecker.AssertClose((plus - minus) / (2 * GradientChecker.EPSILON), (float)analytic[i], $"input gradient [{i}]");
            }
        }

        [Fact]
        public void Network_HasTheExpectedShapes()
        {
            var network = (ConvolutionalNetwork)NetworkFactory.BuildConvolutional(new ConvolutionalNetworkSettings
            {
                InputChannels = 1,
                InputHeight = 28,
                InputWidth = 28,
                Layers =
                {
                    new ConvolutionLayerSettings { Filters = 8, KernelSize = 3, Padding = 1 },
                    new PoolingLayerSettings { Size = 2 },
                    new ConvolutionLayerSettings { Filters = 16, KernelSize = 3 },
                    new PoolingLayerSettings { Size = 2 },
                    new DenseLayerSettings { NeuronsCount = 32 }
                },
                OutputDimension = 10,
                OutputApplySoftMax = true
            });

            Assert.Equal(new[] { new Shape(8, 28, 28), new Shape(8, 14, 14), new Shape(16, 12, 12), new Shape(16, 6, 6), new Shape(32, 1, 1), new Shape(10, 1, 1) },
                         network.Layers.Select(x => x.OutputShape));

            Quad[] outputs = network.Propagate(new Quad[28 * 28]);
            Assert.Equal(10, outputs.Length);
            Assert.Equal(1f, outputs.Sum(x => (float)x), 1e-4f);
        }

        [Fact]
        public void Build_RejectsInvalidLayers()
        {
            ConvolutionalNetworkSettings Settings(params ConvolutionalLayerSettings[] layers)
            {
                var settings = new ConvolutionalNetworkSettings { InputHeight = 5, InputWidth = 5, OutputDimension = 2 };
                foreach (var layer in layers) settings.Layers.Add(layer);
                return settings;
            }

            Assert.ThrowsAny<ArgumentException>(() => NetworkFactory.BuildConvolutional(Settings(new DenseLayerSettings { NeuronsCount = 3 }, new ConvolutionLayerSettings())));
            Assert.ThrowsAny<ArgumentException>(() => NetworkFactory.BuildConvolutional(Settings(new ConvolutionLayerSettings { KernelSize = 6 })));
            Assert.ThrowsAny<ArgumentException>(() => NetworkFactory.BuildConvolutional(Settings(new ConvolutionLayerSettings { KernelSize = 3, Padding = 3 })));
            Assert.ThrowsAny<ArgumentException>(() => NetworkFactory.BuildConvolutional(Settings(new PoolingLayerSettings { Size = 6 })));
            Assert.ThrowsAny<ArgumentException>(() => NetworkFactory.BuildConvolutional(Settings(new ConvolutionLayerSettings { Stride = 0 })));
            Assert.ThrowsAny<ArgumentException>(() => NetworkFactory.BuildConvolutional(new ConvolutionalNetworkSettings { InputHeight = 5, InputWidth = 5 }));
            Assert.ThrowsAny<ArgumentException>(() => NetworkFactory.BuildConvolutional(Settings()).Propagate(new Quad[24]));
        }

        /// <summary>
        /// 6 x 6 images with one horizontal or one vertical bar at a random place, plus noise
        /// </summary>
        internal static (Quad[] Image, int Label) Bar(Random random)
        {
            var image = new Quad[36];
            int label = random.Next(2), position = random.Next(6);
            for (int i = 0; i < 6; i++)
            {
                image[label == 0 ? (position * 6) + i : (i * 6) + position] = 1;
            }

            for (int i = 0; i < image.Length; i++) image[i] += (Quad)((random.NextDouble() - 0.5) * 0.3);
            return (image, label);
        }

        internal static IConvolutionalNetwork BarClassifier(OptimizerTypes optimizer, float learningRate)
        {
            IConvolutionalNetwork network = NetworkFactory.BuildConvolutional(new ConvolutionalNetworkSettings
            {
                InputHeight = 6,
                InputWidth = 6,
                Layers =
                {
                    // LeakyReLU: with unseeded initial weights a plain ReLU filter now and then starts out dead
                    new ConvolutionLayerSettings { Filters = 4, KernelSize = 3, Padding = 1, ActivationType = ActivationTypes.LeakyRELU },
                    new PoolingLayerSettings { Size = 2 }
                },
                OutputDimension = 2,
                OutputApplySoftMax = true,
                OptimizerType = optimizer,
                Seed = 1   // learning tests start from fixed weights, so they cannot fail by bad luck
            });

            foreach (LearningRateOptimizer x in network.Optimizers) x.LearningRate = learningRate;
            return network;
        }

        internal static double BarAccuracy(IConvolutionalNetwork network, Random random, int count = 200)
        {
            int correct = 0;
            for (int i = 0; i < count; i++)
            {
                var (image, label) = Bar(random);
                Quad[] outputs = network.Propagate(image);
                if ((outputs[label] > outputs[1 - label])) correct++;
            }

            return correct / (double)count;
        }

        [Fact]
        public void Network_LearnsToTellHorizontalFromVerticalBars()
        {
            var random = new Random(20);
            IConvolutionalNetwork network = BarClassifier(OptimizerTypes.Normal, 0.05f);

            for (int i = 0; i < 600; i++)
            {
                var (image, label) = Bar(random);
                network.Propagate(image);
                network.Backpropagate(label == 0 ? new Quad[] { 1, 0 } : new Quad[] { 0, 1 });
            }

            Assert.True(BarAccuracy(network, new Random(21)) >= 0.98);
        }
    }
}
