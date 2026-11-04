using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Convolutional;
using ANN.Net.IdentityGenerators;
using Xunit;
using Xunit.Abstractions;

namespace ANN.Net.Tests.Compact
{
    /// <summary>
    /// Pruned compact networks compute what the network computes with the small weights set to 0
    /// </summary>
    public class CompactPruningTests
    {
        private readonly ITestOutputHelper output;

        public CompactPruningTests(ITestOutputHelper output) => this.output = output;

        private static Quad[][] Samples(int width, int count, int seed)
        {
            var random = new Random(seed);
            return Enumerable.Range(0, count).Select(_ => Enumerable.Range(0, width).Select(_ => (Quad)((random.NextDouble() * 4) - 2)).ToArray()).ToArray();
        }

        private static void AssertClose(Func<Quad[], Quad[]> expected, ICompactNetwork actual, Quad[][] samples, float tolerance = 1e-5f)
        {
            foreach (Quad[] sample in samples)
            {
                Quad[] e = expected(sample), a = actual.Propagate(sample);
                Assert.Equal(e.Length, a.Length);
                for (int i = 0; i < e.Length; i++)
                {
                    Assert.True(Math.Abs(e[i] - a[i]) <= tolerance * Math.Max(1, Math.Abs(e[i])), $"output {i}: expected {e[i]}, pruned {a[i]}");
                }
            }
        }

        private static GraphNetworkSettings RandomGraph(int seed)
        {
            var random = new Random(seed);
            var settings = new GraphNetworkSettings { OutputActivationType = ActivationTypes.Sigmoid, HiddenActivationType = ActivationTypes.HyperbolicTangens };
            for (uint i = 0; i < 4; i++) settings.Nodes.Add(new GraphNode(i, NodeType.INPUT));
            settings.Nodes.Add(new GraphNode(4, NodeType.BIAS));
            for (uint i = 0; i < 16; i++) settings.Nodes.Add(new GraphNode(10 + i, NodeType.HIDDEN, i % 3 == 0 ? ActivationTypes.Sigmoid : null));
            for (uint i = 0; i < 3; i++) settings.Nodes.Add(new GraphNode(40 + i, NodeType.OUTPUT));

            uint id = 0;
            var order = settings.Nodes.Select(x => x.Id).ToList();
            for (int c = 0; c < 70; c++)
            {
                int from = random.Next(order.Count - 3);
                int to = random.Next(Math.Max(from + 1, 5), order.Count);
                if (settings.Connections.Any(x => x.FromNode == order[from] && x.ToNode == order[to])) continue;
                settings.Connections.Add(new GraphConnection(id++, order[from], order[to], (float)((random.NextDouble() * 2) - 1)));
            }

            return settings;
        }

        /// <summary>
        /// The same graph with the small weights set to 0, keeping the connections, which keeps when neurons fire.
        /// Biases (the weights from bias nodes) are not pruned
        /// </summary>
        private static GraphNetworkSettings Zeroed(GraphNetworkSettings settings, float threshold)
        {
            return new GraphNetworkSettings
            {
                Nodes = settings.Nodes,
                OutputActivationType = settings.OutputActivationType,
                HiddenActivationType = settings.HiddenActivationType,
                Connections = settings.Connections
                    .Select(x => new GraphConnection(x.Id, x.FromNode, x.ToNode, x.FromNode != 4 && Math.Abs(x.Weight) < threshold ? 0 : x.Weight, x.Enabled))
                    .ToList()
            };
        }

        [Theory]
        [InlineData(1, 0.3f)]
        [InlineData(2, 0.5f)]
        [InlineData(3, 0.8f)]
        [InlineData(4, 2f)]
        public void PrunedGraph_ComputesTheGraphWithoutTheSmallWeights(int seed, float threshold)
        {
            GraphNetworkSettings settings = RandomGraph(seed);
            IGraphNetwork reference = NetworkFactory.BuildGraph(Zeroed(settings, threshold));
            ICompactNetwork full = NetworkCompactor.CompactGraph(settings);
            ICompactNetwork pruned = NetworkCompactor.CompactGraph(settings, new CompactOptions { PruneThreshold = threshold });

            AssertClose(reference.Propagate, pruned, Samples(4, 12, seed));
            Assert.True(pruned.ParameterCount < full.ParameterCount);
            output.WriteLine($"threshold {threshold}: {full.ParameterCount} -> {pruned.ParameterCount} values");
        }

        [Fact]
        public void NeuronWithoutWeights_BecomesAConstantInTheBiasOfItsConsumers()
        {
            var settings = new GraphNetworkSettings
            {
                Nodes =
                {
                    new GraphNode(0, NodeType.INPUT), new GraphNode(1, NodeType.BIAS),
                    new GraphNode(2, NodeType.HIDDEN, ActivationTypes.Sigmoid), new GraphNode(3, NodeType.HIDDEN, ActivationTypes.HyperbolicTangens),
                    new GraphNode(4, NodeType.OUTPUT, ActivationTypes.Identity)
                },
                Connections =
                {
                    new GraphConnection(0, 0, 2, 0.01f), new GraphConnection(1, 1, 2, 0.7f),  // 2 = sigmoid(0.7) once 0.01 is gone
                    new GraphConnection(2, 2, 3, 1.5f), new GraphConnection(3, 1, 3, -0.2f), // 3 = tanh(1.5 * 2 - 0.2), a constant too
                    new GraphConnection(4, 3, 4, 2f), new GraphConnection(5, 0, 4, 3f)
                }
            };

            ICompactNetwork pruned = NetworkCompactor.CompactGraph(settings, new CompactOptions { PruneThreshold = 0.1f });

            // only the output remains: its weight from the input and its bias
            Assert.Equal(2, pruned.ParameterCount);
            float constant = 2 * MathF.Tanh((1.5f / (1 + MathF.Exp(-0.7f))) - 0.2f);
            foreach (float x in new[] { -1f, 0f, 2.5f })
            {
                Assert.Equal((3 * x) + constant, pruned.Propagate(x)[0], 1e-5f);
            }
        }

        [Fact]
        public void PruneEverything_LeavesConstantOutputs()
        {
            GraphNetworkSettings settings = RandomGraph(5);
            IGraphNetwork reference = NetworkFactory.BuildGraph(Zeroed(settings, float.PositiveInfinity));
            ICompactNetwork pruned = NetworkCompactor.CompactGraph(settings, new CompactOptions { PruneFraction = 1 });

            Assert.Equal(3, pruned.ParameterCount); // the bias of every output
            AssertClose(reference.Propagate, pruned, Samples(4, 4, 5));
        }

        [Fact]
        public void PruneFraction_RemovesThatShareOfTheWeights()
        {
            IMatrixNetwork network = NetworkFactory.BuildMatrixFFN(new NetworkSettings
            {
                Seed = 8,
                IdentityGenerator = new IncrementalGenerator(),
                InputSettings = new InputLayerSettings { NeuronsCount = 30, HasBiasNeuron = true },
                OutputSettings = new OutputLayerSettings { NeuronsCount = 5, ActivationType = ActivationTypes.Sigmoid, InitializationType = WeightInitType.Xavier }
            },
            new RecurrentLayerSettings
            {
                NeuronsCount = 40, LayersCount = 1, CellType = NeuronType.Normal, PrevLayerConnectionType = LayerConnectionType.FullyConnected,
                ActivationType = ActivationTypes.RELU, HasBiasNeuron = true, InitializationType = WeightInitType.Xavier
            });

            const int weights = (30 * 40) + (40 * 5), biases = 40 + 5;
            ICompactNetwork pruned = NetworkCompactor.Compact(network, new CompactOptions { PruneFraction = 0.75 });

            // a quarter of the weights is left; a matrix that is still too dense for sparse storage keeps its zeros
            Assert.InRange(pruned.ParameterCount, 1, (weights * 0.4) + biases);
            Assert.True(pruned.SizeInBytes < NetworkCompactor.Compact(network).SizeInBytes * 0.6);
        }

        [Fact]
        public void TrainedClassifier_KeepsItsAccuracyWhenMostWeightsArePruned()
        {
            const int features = 10, classes = 4;
            var centers = new Random(100);
            float[][] center = Enumerable.Range(0, classes).Select(_ => Enumerable.Range(0, features).Select(_ => (float)((centers.NextDouble() * 2) - 1)).ToArray()).ToArray();
            (Quad[] Input, int Label)[] Data(int count, int seed)
            {
                var random = new Random(seed);
                return Enumerable.Range(0, count).Select(i => (center[i % classes].Select(c => (Quad)(c + (((random.NextDouble() + random.NextDouble() + random.NextDouble()) - 1.5) * 0.9))).ToArray(), i % classes)).ToArray();
            }

            IMatrixNetwork network = NetworkFactory.BuildMatrixFFN(new NetworkSettings
            {
                Seed = 3,
                IdentityGenerator = new IncrementalGenerator(),
                InputSettings = new InputLayerSettings { NeuronsCount = features, HasBiasNeuron = true },
                OutputSettings = new OutputLayerSettings { NeuronsCount = classes, ActivationType = ActivationTypes.Sigmoid, OptimizerType = OptimizerTypes.Adam, InitializationType = WeightInitType.Xavier }
            },
            new RecurrentLayerSettings
            {
                NeuronsCount = 48, LayersCount = 2, CellType = NeuronType.Normal, PrevLayerConnectionType = LayerConnectionType.FullyConnected,
                ActivationType = ActivationTypes.RELU, HasBiasNeuron = true, OptimizerType = OptimizerTypes.Adam, InitializationType = WeightInitType.Xavier
            });
            foreach (LearningRateOptimizer optimizer in network.Optimizers) optimizer.LearningRate = 0.003f;
            foreach (var (input, label) in Enumerable.Range(0, 8).SelectMany(_ => Data(800, 1)))
            {
                network.Propagate(input);
                network.Backpropagate(Enumerable.Range(0, classes).Select(c => (Quad)(c == label ? 1 : 0)).ToArray());
            }

            var test = Data(1000, 2);
            double Accuracy(Func<Quad[], Quad[]> propagate) => test.Count(x => { Quad[] o = propagate(x.Input); return Array.IndexOf(o, o.Max()) == x.Label; }) / (double)test.Length;
            double reference = Accuracy(network.Propagate);
            long size = NetworkCompactor.Compact(network).SizeInBytes;

            // without retraining, global magnitude pruning holds up to about 80 % here; 90 % breaks the classifier
            foreach (double fraction in new[] { 0.5, 0.7, 0.8, 0.9 })
            {
                foreach (WeightPrecision precision in new[] { WeightPrecision.Single, WeightPrecision.Int8 })
                {
                    ICompactNetwork pruned = NetworkCompactor.Compact(network, new CompactOptions { PruneFraction = fraction, Precision = precision });
                    double accuracy = Accuracy(pruned.Propagate);
                    output.WriteLine($"prune {fraction:P0} {precision}: accuracy {accuracy:P2} (network {reference:P2}), {pruned.SizeInBytes:N0} bytes instead of {size:N0}, {pruned.ParameterCount:N0} values");
                    if (fraction <= 0.8) Assert.True(reference - accuracy <= 0.015, $"pruning {fraction:P0} {precision}: accuracy {accuracy:P2} instead of {reference:P2}");
                }
            }
        }

        [Fact]
        public void PrunedConvolutionalNetwork_ComputesTheNetworkWithoutTheSmallWeights()
        {
            var settings = new ConvolutionalNetworkSettings
            {
                InputChannels = 2, InputHeight = 10, InputWidth = 10, OutputDimension = 4, Seed = 9,
                Layers =
                {
                    new ConvolutionLayerSettings { Filters = 6, KernelSize = 3, Padding = 1, ActivationType = ActivationTypes.RELU },
                    new PoolingLayerSettings { Type = PoolingType.Max, Size = 2 },
                    new ResidualBlockSettings { Filters = 8, Stride = 1 },
                    new DenseLayerSettings { NeuronsCount = 12, ActivationType = ActivationTypes.HyperbolicTangens }
                }
            };

            var network = (ConvolutionalNetwork)NetworkFactory.BuildConvolutional(settings);
            network.Training = false;
            var reference = (ConvolutionalNetwork)NetworkFactory.BuildConvolutional(settings);
            reference.Training = false;

            const float threshold = 0.1f;
            ICompactNetwork pruned = NetworkCompactor.Compact(network, new CompactOptions { PruneThreshold = threshold });
            foreach (Parameters weights in WeightsOf(reference.Layers))
            {
                for (int i = 0; i < weights.Values.Length; i++)
                {
                    if (Math.Abs(weights.Values[i]) < threshold) weights.Values[i] = 0;
                }
            }

            // batch normalizations of the residual block are folded into the convolutions before pruning, which scales
            // their weights, so the two networks prune slightly different weights
            ICompactNetwork full = NetworkCompactor.Compact(network);
            Quad[][] images = Samples(200, 4, 10);
            Assert.True(pruned.SizeInBytes <= full.SizeInBytes);
            Assert.Contains(images, x => full.Propagate(x).Zip(pruned.Propagate(x), (a, b) => Math.Abs(a - b)).Max() > 1e-3);
            AssertClose(NetworkCompactor.Compact(reference).Propagate, pruned, images, 0.05f);
        }

        private static IEnumerable<Parameters> WeightsOf(IEnumerable<IConvolutionalLayer> layers)
        {
            foreach (IConvolutionalLayer layer in layers)
            {
                switch (layer)
                {
                    case ConvolutionLayer convolution:
                        yield return convolution.Weights;
                        break;
                    case DenseLayer dense:
                        yield return dense.Weights;
                        break;
                    case ResidualBlockLayer residual:
                        foreach (Parameters inner in WeightsOf(residual.Main.Concat(residual.Shortcut))) yield return inner;
                        break;
                }
            }
        }
    }
}
