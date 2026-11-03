using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Convolutional;
using ANN.Net.Tests.Networks;
using Xunit;

namespace ANN.Net.Tests.Convolutional
{
    /// <summary>
    /// ResNet-18 (He et al., 2016) built from convolution, batch normalization, pooling and residual block layers
    /// </summary>
    public class ResNetTests
    {
        /// <summary>
        /// ResNet-18 as in torchvision: 7x7/2 convolution, batch normalization + ReLU, 3x3/2 max pooling, four stages of two
        /// basic residual blocks with 64, 128, 256 and 512 filters (the first block of every later stage halves the size),
        /// global average pooling and the fully connected classifier. <paramref name="width"/> replaces the 64 base filters
        /// </summary>
        internal static ConvolutionalNetworkSettings ResNet18(int height, int width, int classes, int baseFilters = 64, OptimizerTypes optimizer = OptimizerTypes.Normal)
        {
            var settings = new ConvolutionalNetworkSettings
            {
                InputChannels = 3,
                InputHeight = height,
                InputWidth = width,
                OutputDimension = classes,
                OutputApplySoftMax = true,
                OptimizerType = optimizer,
                Seed = 1,
                Layers =
                {
                    new ConvolutionLayerSettings { Filters = baseFilters, KernelSize = 7, Stride = 2, Padding = 3, UseBias = false, ActivationType = ActivationTypes.Identity },
                    new BatchNormalizationLayerSettings { ActivationType = ActivationTypes.RELU },
                    new PoolingLayerSettings { Type = PoolingType.Max, Size = 3, Stride = 2, Padding = 1 }
                }
            };

            foreach (int stage in new[] { 1, 2, 4, 8 })
            {
                settings.Layers.Add(new ResidualBlockSettings { Filters = baseFilters * stage, Stride = stage == 1 ? 1 : 2 });
                settings.Layers.Add(new ResidualBlockSettings { Filters = baseFilters * stage, Stride = 1 });
            }

            settings.Layers.Add(new GlobalAveragePoolingLayerSettings());
            return settings;
        }

        [Fact]
        public void ResNet18_HasTheArchitectureAndParameterCountOfTorchvision()
        {
            var network = (ConvolutionalNetwork)NetworkFactory.BuildConvolutional(ResNet18(224, 224, 1000));

            Assert.Equal(new[]
            {
                new Shape(64, 112, 112),                          // conv1
                new Shape(64, 112, 112),                          // bn1 + relu
                new Shape(64, 56, 56),                            // maxpool
                new Shape(64, 56, 56), new Shape(64, 56, 56),     // layer1
                new Shape(128, 28, 28), new Shape(128, 28, 28),   // layer2
                new Shape(256, 14, 14), new Shape(256, 14, 14),   // layer3
                new Shape(512, 7, 7), new Shape(512, 7, 7),       // layer4
                new Shape(512, 1, 1),                             // avgpool
                new Shape(1000, 1, 1)                             // fc
            }, network.Layers.Select(x => x.OutputShape));

            // the first block of layer2, layer3 and layer4 projects its input with a 1x1 convolution, every other one adds it
            ResidualBlockLayer[] blocks = network.Layers.OfType<ResidualBlockLayer>().ToArray();
            Assert.Equal(new[] { false, false, true, false, true, false, true, false }, blocks.Select(x => x.Shortcut.Count > 0));
            Assert.All(blocks.SelectMany(x => x.Main.Concat(x.Shortcut)).OfType<ConvolutionLayer>(), x => Assert.Null(x.Biases));

            // torchvision.models.resnet18(): sum(p.numel() for p in model.parameters()) == 11_689_512
            Assert.Equal(11_689_512, network.Optimizers.Count);
        }

        private static Quad[] Image(Random random, int size) => Enumerable.Range(0, size).Select(_ => (Quad)((random.NextDouble() * 2) - 1)).ToArray();

        /// <summary>
        /// ResNet-18 with 8 instead of 64 base filters on 32 x 32 images: the same 18 layers, small enough to run in tests,
        /// trained a few steps so that the weights and the running statistics are no longer at their initial values
        /// </summary>
        private static IConvolutionalNetwork TrainedNarrowResNet18()
        {
            IConvolutionalNetwork network = NetworkFactory.BuildConvolutional(ResNet18(32, 32, 10, baseFilters: 8, OptimizerTypes.Adam));
            foreach (LearningRateOptimizer optimizer in network.Optimizers) optimizer.LearningRate = 0.001f;

            var random = new Random(2);
            for (int i = 0; i < 8; i++)
            {
                network.Propagate(Image(random, 3 * 32 * 32));
                network.Backpropagate(Enumerable.Range(0, 10).Select(c => (Quad)(c == i ? 1 : 0)).ToArray());
            }

            return network;
        }

        [Fact]
        public void NarrowResNet18_ExportsToOnnxWithTheSameInferenceOutputs()
        {
            IConvolutionalNetwork network = TrainedNarrowResNet18();
            network.Training = false;
            var random = new Random(3);
            float[][] images = Enumerable.Range(0, 3).Select(_ => Image(random, 3 * 32 * 32).Select(x => (float)x).ToArray()).ToArray();

            float[][] actual = OnnxExportTests.Run(OnnxExporter.Export(network), images);

            for (int i = 0; i < images.Length; i++)
            {
                float[] expected = network.Propagate(images[i].Select(x => (Quad)x).ToArray()).Select(x => (float)x).ToArray();
                for (int j = 0; j < expected.Length; j++) Assert.Equal(expected[j], actual[i][j], 1e-4f);
            }
        }

        [Fact]
        public void NarrowResNet18_RoundTripKeepsWeightsRunningStatisticsAndMode()
        {
            IConvolutionalNetwork original = TrainedNarrowResNet18();

            using var stream = new MemoryStream();
            NetworkSerializer.Save(original, stream);
            stream.Position = 0;
            IConvolutionalNetwork loaded = NetworkSerializer.Load<IConvolutionalNetwork>(stream);

            var random = new Random(4);
            Quad[] image = Image(random, 3 * 32 * 32);
            Assert.True(loaded.Training);

            original.Training = loaded.Training = false;
            Assert.Equal(original.Propagate(image).Select(x => (float)x), loaded.Propagate(image).Select(x => (float)x));

            original.Training = loaded.Training = true;
            Assert.Equal(original.Propagate(image).Select(x => (float)x), loaded.Propagate(image).Select(x => (float)x));
        }

        [Fact]
        public void Training_UsesImageStatisticsAndInferenceTheRunningAverages()
        {
            IConvolutionalNetwork network = TrainedNarrowResNet18();
            var normalization = ((ConvolutionalNetwork)network).Layers.OfType<BatchNormalizationLayer>().First();
            Quad[] image = Image(new Random(5), 3 * 32 * 32);

            network.Training = false;
            float[] mean = normalization.RunningMean.Select(x => (float)x).ToArray();
            float[] first = network.Propagate(image).Select(x => (float)x).ToArray();
            Assert.Equal(first, network.Propagate(image).Select(x => (float)x));
            Assert.Equal(mean, normalization.RunningMean.Select(x => (float)x));

            network.Training = true;
            float[] training = network.Propagate(image).Select(x => (float)x).ToArray();
            Assert.NotEqual(mean, normalization.RunningMean.Select(x => (float)x));
            Assert.NotEqual(first, training);
        }

        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public void BatchNormalization_GradientsMatchFiniteDifferences(bool training)
        {
            var random = new Random(training ? 6 : 7);
            var optimizers = new List<LearningRateOptimizer>();
            var mode = new NetworkMode { Training = training };
            var layer = new BatchNormalizationLayer(new Shape(3, 4, 5), ActivationTypes.HyperbolicTangens, 0.1f, 1e-5f, mode, OptimizerTypes.Normal, optimizers);
            for (int c = 0; c < 3; c++)
            {
                layer.Gamma.Values[c] = 0.5f + c;
                layer.Beta.Values[c] = 0.2f * c;
                layer.RunningMean[c] = 0.1f * c;
                layer.RunningVariance[c] = 0.5f + c;
            }

            ConvolutionalNetworkTests.AssertInputGradient(layer, random);
            ConvolutionalNetworkTests.AssertParameterGradients(layer, layer.Gamma, optimizers, random);
            ConvolutionalNetworkTests.AssertParameterGradients(layer, layer.Beta, optimizers, random);
        }

        [Fact]
        public void BatchNormalization_NormalizesEveryChannelInTrainingMode()
        {
            var layer = new BatchNormalizationLayer(new Shape(2, 3, 3), ActivationTypes.Identity, 0.1f, 1e-5f, new NetworkMode(), OptimizerTypes.Normal, new List<LearningRateOptimizer>());
            Quad[] input = Image(new Random(8), 18).Select((x, i) => i < 9 ? (x * 3) + 5 : x - 2).ToArray();

            Quad[] output = layer.Propagate(input);

            for (int c = 0; c < 2; c++)
            {
                float[] channel = output.Skip(c * 9).Take(9).Select(x => (float)x).ToArray();
                Assert.Equal(0, channel.Average(), 4);
                Assert.Equal(1, channel.Select(x => x * x).Average(), 3);
            }
        }

        [Theory]
        [InlineData(4, 1)]
        [InlineData(6, 2)]
        public void ResidualBlock_GradientsMatchFiniteDifferences(int filters, int stride)
        {
            var random = new Random(filters * 10 + stride);
            var optimizers = new List<LearningRateOptimizer>();
            var block = new ResidualBlockLayer(new Shape(4, 6, 5), filters, stride, ActivationTypes.HyperbolicTangens, new NetworkMode(),
                                               OptimizerTypes.Normal, WeightInitType.Xavier, optimizers);
            Assert.Equal(filters != 4 || stride != 1, block.Shortcut.Count > 0);

            ConvolutionalNetworkTests.AssertInputGradient(block, random);
            ConvolutionalNetworkTests.AssertParameterGradients(block, ((ConvolutionLayer)block.Main[0]).Weights, optimizers, random);
            ConvolutionalNetworkTests.AssertParameterGradients(block, ((BatchNormalizationLayer)block.Main[3]).Gamma, optimizers, random);
            if (block.Shortcut.Count > 0)
            {
                ConvolutionalNetworkTests.AssertParameterGradients(block, ((ConvolutionLayer)block.Shortcut[0]).Weights, optimizers, random);
            }
        }

        [Theory]
        [InlineData(PoolingType.Max)]
        [InlineData(PoolingType.Average)]
        public void PaddedPooling_IgnoresThePaddingAndMatchesFiniteDifferences(PoolingType type)
        {
            var layer = new PoolingLayer(new Shape(1, 2, 2), type, 3, 2, 1);
            Quad[] output = layer.Propagate(new Quad[] { -1, -2, -3, -4 });

            // one window covers all four values: zeros in the padding would win a max and dilute an average
            Assert.Equal(type == PoolingType.Max ? -1f : -2.5f, (float)Assert.Single(output));

            ConvolutionalNetworkTests.AssertInputGradient(new PoolingLayer(new Shape(2, 7, 6), type, 3, 2, 1), new Random(9), distinct: type == PoolingType.Max);
        }

        [Fact]
        public void GlobalAveragePooling_AveragesEveryChannel()
        {
            var layer = new GlobalAveragePoolingLayer(new Shape(2, 2, 3));

            Assert.Equal(new[] { 3.5f, -1f }, layer.Propagate(new Quad[] { 1, 2, 3, 4, 5, 6, -1, -1, -1, -1, -1, -1 }).Select(x => (float)x));
            ConvolutionalNetworkTests.AssertInputGradient(new GlobalAveragePoolingLayer(new Shape(3, 4, 2)), new Random(10));
        }

        [Fact]
        public void SmallResNet_LearnsToTellHorizontalFromVerticalBars()
        {
            IConvolutionalNetwork network = NetworkFactory.BuildConvolutional(new ConvolutionalNetworkSettings
            {
                InputHeight = 6,
                InputWidth = 6,
                OutputDimension = 2,
                OutputApplySoftMax = true,
                OptimizerType = OptimizerTypes.Adam,
                Seed = 1,
                Layers =
                {
                    new ConvolutionLayerSettings { Filters = 4, KernelSize = 3, Padding = 1, UseBias = false, ActivationType = ActivationTypes.Identity },
                    new BatchNormalizationLayerSettings { ActivationType = ActivationTypes.RELU },
                    new ResidualBlockSettings { Filters = 4 },
                    new ResidualBlockSettings { Filters = 8, Stride = 2 },
                    new GlobalAveragePoolingLayerSettings()
                }
            });
            foreach (LearningRateOptimizer optimizer in network.Optimizers) optimizer.LearningRate = 0.01f;

            var random = new Random(11);
            for (int i = 0; i < 600; i++)
            {
                var (image, label) = ConvolutionalNetworkTests.Bar(random);
                network.Propagate(image);
                network.Backpropagate(label == 0 ? new Quad[] { 1, 0 } : new Quad[] { 0, 1 });
            }

            network.Training = false;
            double accuracy = ConvolutionalNetworkTests.BarAccuracy(network, new Random(12));
            Assert.True(accuracy >= 0.95, $"accuracy {accuracy:P0}");
        }
    }
}
