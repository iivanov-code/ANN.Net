using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.IdentityGenerators;
using ANN.Net.Tests.Networks;
using Xunit;
using Xunit.Abstractions;

namespace ANN.Net.Tests.Compact
{
    /// <summary>
    /// 16-bit and 8-bit weights: how much they change the outputs and the accuracy of trained networks, and ONNX
    /// files with the same weights
    /// </summary>
    public class CompactPrecisionTests
    {
        private readonly ITestOutputHelper output;

        public CompactPrecisionTests(ITestOutputHelper output) => this.output = output;

        private const int Features = 10, Classes = 4;

        /// <summary>
        /// Overlapping clusters, one per class, with noise
        /// </summary>
        private static (Quad[] Input, int Label)[] Data(int count, int seed)
        {
            var centers = new Random(100);
            float[][] center = Enumerable.Range(0, Classes).Select(_ => Enumerable.Range(0, Features).Select(_ => (float)((centers.NextDouble() * 2) - 1)).ToArray()).ToArray();

            var random = new Random(seed);
            return Enumerable.Range(0, count).Select(i =>
            {
                int label = i % Classes;
                Quad[] input = center[label].Select(c => (Quad)(c + (((random.NextDouble() + random.NextDouble() + random.NextDouble()) - 1.5) * 0.9))).ToArray();
                return (input, label);
            }).ToArray();
        }

        private static IMatrixNetwork TrainedClassifier()
        {
            IMatrixNetwork network = NetworkFactory.BuildMatrixFFN(new NetworkSettings
            {
                Seed = 3,
                IdentityGenerator = new IncrementalGenerator(),
                LossType = LossFunctionTypes.Squared,
                InputSettings = new InputLayerSettings { NeuronsCount = Features, HasBiasNeuron = true },
                OutputSettings = new OutputLayerSettings { NeuronsCount = Classes, ActivationType = ActivationTypes.Sigmoid, OptimizerType = OptimizerTypes.Adam, InitializationType = WeightInitType.Xavier }
            },
            new RecurrentLayerSettings
            {
                NeuronsCount = 48, LayersCount = 2, CellType = NeuronType.Normal, PrevLayerConnectionType = LayerConnectionType.FullyConnected,
                ActivationType = ActivationTypes.RELU, HasBiasNeuron = true, OptimizerType = OptimizerTypes.Adam, InitializationType = WeightInitType.Xavier
            });
            foreach (LearningRateOptimizer optimizer in network.Optimizers) optimizer.LearningRate = 0.003f;

            var data = Data(800, 1);
            for (int epoch = 0; epoch < 8; epoch++)
            {
                foreach (var (input, label) in data)
                {
                    network.Propagate(input);
                    network.Backpropagate(Enumerable.Range(0, Classes).Select(c => (Quad)(c == label ? 1 : 0)).ToArray());
                }
            }

            return network;
        }

        private static int ArgMax(Quad[] values) => Array.IndexOf(values, values.Max());

        private static double Accuracy(Func<Quad[], Quad[]> propagate, (Quad[] Input, int Label)[] data) =>
            data.Count(x => ArgMax(propagate(x.Input)) == x.Label) / (double)data.Length;

        [Fact]
        public void TrainedClassifier_KeepsItsAccuracyWithHalfAndInt8Weights()
        {
            IMatrixNetwork network = TrainedClassifier();
            var test = Data(1000, 2);
            double reference = Accuracy(network.Propagate, test);
            Assert.True(reference > 0.8, $"the classifier did not learn: {reference:P1}");

            ICompactNetwork single = NetworkCompactor.Compact(network);
            foreach (WeightPrecision precision in new[] { WeightPrecision.Single, WeightPrecision.Half, WeightPrecision.Int8 })
            {
                ICompactNetwork compact = NetworkCompactor.Compact(network, new CompactOptions { Precision = precision });
                double accuracy = Accuracy(compact.Propagate, test);
                double largest = test.Max(x => network.Propagate(x.Input).Zip(compact.Propagate(x.Input), (a, b) => Math.Abs(a - b)).Max());
                output.WriteLine($"{precision}: accuracy {accuracy:P2} (network {reference:P2}), largest output difference {largest:G3}, {compact.SizeInBytes:N0} bytes");

                Assert.Equal(precision, compact.Precision);
                Assert.True(Math.Abs(accuracy - reference) <= 0.01, $"{precision}: accuracy {accuracy:P2} instead of {reference:P2}");
                Assert.True(largest <= (precision == WeightPrecision.Single ? 1e-5 : precision == WeightPrecision.Half ? 5e-3 : 5e-2), $"{precision}: outputs differ by {largest}");
            }

            long weights = (10 * 48) + (48 * 48) + (48 * 4);   // the biases are kept apart
            long others = single.SizeInBytes - (weights * 4);
            Assert.Equal(others + (weights * 2), NetworkCompactor.Compact(network, new CompactOptions { Precision = WeightPrecision.Half }).SizeInBytes);
        }

        [Theory]
        [InlineData(WeightPrecision.Half)]
        [InlineData(WeightPrecision.Int8)]
        public void OnnxExport_ComputesWhatTheCompactNetworkOfTheSamePrecisionComputes(WeightPrecision precision)
        {
            IMatrixNetwork network = TrainedClassifier();
            var samples = Data(20, 3);
            ICompactNetwork compact = NetworkCompactor.Compact(network, new CompactOptions { Precision = precision });
            byte[] model = OnnxExporter.Export(network, new OnnxExportOptions { WeightPrecision = precision });
            float[][] onnx = OnnxExportTests.Run(model, samples.Select(x => x.Input.Select(v => (float)v).ToArray()).ToArray(), optimize: false);

            for (int i = 0; i < samples.Length; i++)
            {
                Quad[] expected = compact.Propagate(samples[i].Input);
                for (int j = 0; j < expected.Length; j++) Assert.Equal(expected[j], onnx[i][j], 1e-5f);
            }

            output.WriteLine($"{precision}: {model.Length:N0} bytes instead of {OnnxExporter.Export(network).Length:N0}");
            Assert.True(model.Length < OnnxExporter.Export(network).Length * (precision == WeightPrecision.Half ? 0.65 : 0.5)); // the graph itself takes a share of a small model
        }

        [Fact]
        public void GraphNetwork_WithHalfAndInt8Weights_StaysClose()
        {
            var random = new Random(5);
            var settings = new GraphNetworkSettings
            {
                Nodes = { new GraphNode(0, NodeType.INPUT), new GraphNode(1, NodeType.INPUT), new GraphNode(2, NodeType.BIAS), new GraphNode(9, NodeType.OUTPUT) }
            };
            uint id = 0;
            for (uint h = 0; h < 20; h++)
            {
                settings.Nodes.Add(new GraphNode(10 + h, NodeType.HIDDEN, h % 2 == 0 ? ActivationTypes.HyperbolicTangens : ActivationTypes.RELU));
                settings.Connections.Add(new GraphConnection(id++, h % 2, 10 + h, (float)((random.NextDouble() * 4) - 2)));
                settings.Connections.Add(new GraphConnection(id++, 2, 10 + h, (float)((random.NextDouble() * 2) - 1)));
                if (h > 0) settings.Connections.Add(new GraphConnection(id++, 9 + h, 10 + h, (float)((random.NextDouble() * 2) - 1)));
                settings.Connections.Add(new GraphConnection(id++, 10 + h, 9, (float)((random.NextDouble() * 2) - 1)));
            }

            IGraphNetwork network = NetworkFactory.BuildGraph(settings);
            foreach (WeightPrecision precision in new[] { WeightPrecision.Half, WeightPrecision.Int8 })
            {
                ICompactNetwork compact = NetworkCompactor.CompactGraph(settings, new CompactOptions { Precision = precision });
                float[][] onnx = OnnxExportTests.Run(OnnxExporter.Export(network, new OnnxExportOptions { WeightPrecision = precision }), new[] { new[] { 0.3f, -0.7f }, new[] { 1.2f, 0.4f } }, optimize: false);
                for (int s = 0; s < 2; s++)
                {
                    Quad[] input = s == 0 ? new Quad[] { 0.3f, -0.7f } : new Quad[] { 1.2f, 0.4f };
                    float expected = network.Propagate(input)[0];
                    float actual = compact.Propagate(input)[0];
                    Assert.True(Math.Abs(expected - actual) <= (precision == WeightPrecision.Half ? 0.01 : 0.1), $"{precision}: {expected} became {actual}");
                    Assert.Equal(actual, onnx[s][0], 1e-5f);
                }
            }
        }

        [Fact]
        public void ConvolutionalNetwork_WithHalfAndInt8Weights_StaysClose()
        {
            IConvolutionalNetwork network = NetworkFactory.BuildConvolutional(new ConvolutionalNetworkSettings
            {
                InputChannels = 3, InputHeight = 12, InputWidth = 12, OutputDimension = 5, Seed = 4,
                Layers =
                {
                    new ConvolutionLayerSettings { Filters = 8, KernelSize = 3, Padding = 1, UseBias = false, ActivationType = ActivationTypes.Identity },
                    new BatchNormalizationLayerSettings { ActivationType = ActivationTypes.RELU },
                    new ResidualBlockSettings { Filters = 12, Stride = 2 },
                    new GlobalAveragePoolingLayerSettings(),
                    new DenseLayerSettings { NeuronsCount = 10, ActivationType = ActivationTypes.HyperbolicTangens }
                }
            });
            network.Training = false;

            var random = new Random(6);
            Quad[][] images = Enumerable.Range(0, 5).Select(_ => Enumerable.Range(0, 3 * 12 * 12).Select(_ => (Quad)((random.NextDouble() * 2) - 1)).ToArray()).ToArray();
            ICompactNetwork single = NetworkCompactor.Compact(network);
            foreach (WeightPrecision precision in new[] { WeightPrecision.Half, WeightPrecision.Int8 })
            {
                ICompactNetwork compact = NetworkCompactor.Compact(network, new CompactOptions { Precision = precision });
                float[][] onnx = OnnxExportTests.Run(OnnxExporter.Export(network, new OnnxExportOptions { WeightPrecision = precision }),
                                                     images.Select(x => x.Select(v => (float)v).ToArray()).ToArray());
                double tolerance = precision == WeightPrecision.Half ? 5e-3 : 5e-2;
                for (int i = 0; i < images.Length; i++)
                {
                    Quad[] expected = single.Propagate(images[i]);
                    Quad[] actual = compact.Propagate(images[i]);
                    for (int j = 0; j < expected.Length; j++)
                    {
                        Assert.True(Math.Abs(expected[j] - actual[j]) <= tolerance, $"{precision}: {expected[j]} became {actual[j]}");
                        Assert.True(Math.Abs(expected[j] - onnx[i][j]) <= tolerance, $"{precision} ONNX: {expected[j]} became {onnx[i][j]}");
                    }
                }
            }
        }
    }
}
