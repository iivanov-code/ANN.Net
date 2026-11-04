using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Compact;
using ANN.Net.Convolutional;
using ANN.Net.IdentityGenerators;
using ANN.Net.Tests.Convolutional;
using ANN.Net.Tests.Networks;
using Xunit;

namespace ANN.Net.Tests.Compact
{
    /// <summary>
    /// <see cref="NetworkCompactor"/> replaces the objects of a network by flat arrays and must still compute the same values
    /// </summary>
    public class CompactNetworkTests
    {
        private static Quad[][] Samples(int width, int count = 6, int seed = 1)
        {
            var random = new Random(seed);
            return Enumerable.Range(0, count).Select(_ => Enumerable.Range(0, width).Select(_ => (Quad)((random.NextDouble() * 4) - 2)).ToArray()).ToArray();
        }

        private static void AssertSameOutputs(INetwork network, ICompactNetwork compact, Quad[][] samples)
        {
            foreach (Quad[] sample in samples)
            {
                float[] expected = network.Propagate(sample).Select(x => (float)x).ToArray();
                float[] actual = compact.Propagate(sample).Select(x => (float)x).ToArray();
                Assert.Equal(expected.Length, actual.Length);

                // dividing by a sum of outputs close to zero magnifies rounding by Σ|outputs|
                float condition = Math.Max(1f, expected.Sum(Math.Abs));
                for (int i = 0; i < expected.Length; i++)
                {
                    Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5f * Math.Max(1f, Math.Abs(expected[i])) * condition,
                                $"output {i}: network {expected[i]}, compact {actual[i]}");
                }
            }
        }

        private static INetwork Layered(bool normalize, bool softMax)
        {
            return NetworkFactory.BuildFFN(new NetworkSettings
            {
                IdentityGenerator = new IncrementalGenerator(),
                Seed = 1,
                ShouldNormalize = normalize,
                MinMaxValues = normalize ? Enumerable.Range(0, 4).Select(i => Tuple.Create<Quad, Quad>(-3 - i, 2 + i)).ToList() : null,
                OutputApplySoftMax = softMax,
                InputSettings = new InputLayerSettings { NeuronsCount = 4, ActivationType = normalize ? ActivationTypes.Sigmoid : ActivationTypes.Identity, HasBiasNeuron = true },
                OutputSettings = new OutputLayerSettings { NeuronsCount = 3, ActivationType = ActivationTypes.HyperbolicTangens }
            },
            new RecurrentLayerSettings { NeuronsCount = 6, LayersCount = 2, CellType = NeuronType.Normal, ActivationType = ActivationTypes.RELU, HasBiasNeuron = true });
        }

        [Theory]
        [InlineData(false, false)]
        [InlineData(true, false)]
        [InlineData(true, true)]
        public void LayeredNetwork_GivesTheSameOutputs(bool normalize, bool softMax)
        {
            INetwork network = Layered(normalize, softMax);
            ICompactNetwork compact = NetworkCompactor.Compact(network);

            Assert.Equal(4, compact.InputCount);
            Assert.Equal(3, compact.OutputCount);
            Assert.Equal((5 * 6) + (7 * 6) + (7 * 3) + (normalize ? 8 : 0), compact.ParameterCount);
            AssertSameOutputs(network, compact, Samples(4));
        }

        [Fact]
        public void GraphNetwork_WithSkipConnectionsAndEveryActivation_GivesTheSameOutputs()
        {
            var settings = new GraphNetworkSettings
            {
                Nodes = { new GraphNode(0, NodeType.INPUT), new GraphNode(1, NodeType.INPUT, ActivationTypes.ArcusTangens), new GraphNode(2, NodeType.BIAS) },
                OutputActivationType = ActivationTypes.Identity
            };

            uint id = 0;
            var random = new Random(2);
            float Weight() => (float)((random.NextDouble() * 2) - 1);
            for (int i = 0; i < ActivationFunctionTests.Buildable.Length; i++)
            {
                uint hidden = (uint)(10 + i);
                settings.Nodes.Add(new GraphNode(hidden, NodeType.HIDDEN, ActivationFunctionTests.Buildable[i]));
                settings.Connections.Add(new GraphConnection(id++, (uint)(i % 2), hidden, Weight()));
                settings.Connections.Add(new GraphConnection(id++, 2, hidden, Weight()));
                if (i > 0) settings.Connections.Add(new GraphConnection(id++, hidden - 1, hidden, Weight()));   // a deep chain
            }

            settings.Nodes.Add(new GraphNode(100, NodeType.OUTPUT));
            settings.Nodes.Add(new GraphNode(101, NodeType.OUTPUT, ActivationTypes.Sigmoid));
            settings.Nodes.Add(new GraphNode(102, NodeType.OUTPUT));                                            // never fires: 0
            settings.Connections.Add(new GraphConnection(id++, 23, 100, Weight()));
            settings.Connections.Add(new GraphConnection(id++, 0, 100, Weight()));                              // skip connection
            settings.Connections.Add(new GraphConnection(id++, 15, 101, Weight()));
            settings.Connections.Add(new GraphConnection(id, 2, 101, Weight()));
            IGraphNetwork network = NetworkFactory.BuildGraph(settings);

            AssertSameOutputs(network, NetworkCompactor.Compact(network), Samples(2, 10));
        }

        /// <summary>
        /// Random acyclic graph with dead branches, disabled connections, unconnected outputs and bias nodes
        /// </summary>
        private static GraphNetworkSettings RandomGraph(int seed)
        {
            var random = new Random(seed);
            ActivationTypes[] activations = ActivationFunctionTests.Buildable;
            var settings = new GraphNetworkSettings
            {
                OutputApplySoftMax = seed % 3 == 0,
                OutputActivationType = ActivationTypes.Sigmoid,
                InputActivationType = seed % 2 == 0 ? ActivationTypes.Identity : ActivationTypes.HyperbolicTangens
            };

            var order = new List<uint>();
            for (uint i = 0; i < 3; i++) { settings.Nodes.Add(new GraphNode(i, NodeType.INPUT)); order.Add(i); }
            settings.Nodes.Add(new GraphNode(3, NodeType.BIAS)); order.Add(3);
            for (uint i = 0; i < 12; i++)
            {
                settings.Nodes.Add(new GraphNode(10 + i, NodeType.HIDDEN, random.Next(3) == 0 ? null : activations[random.Next(activations.Length)]));
                order.Add(10 + i);
            }

            for (uint i = 0; i < 3; i++) { settings.Nodes.Add(new GraphNode(50 + i, NodeType.OUTPUT)); order.Add(50 + i); }

            // connections only go forward in this order, so the graph is acyclic; hidden nodes may end up dead
            uint id = 100;
            for (int c = 0; c < 40; c++)
            {
                int from = random.Next(order.Count - 3);
                int to = random.Next(Math.Max(from + 1, 4), order.Count);
                settings.Connections.Add(new GraphConnection(id++, order[from], order[to], (float)((random.NextDouble() * 4) - 2), random.Next(8) != 0));
            }

            return settings;
        }

        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(3)]
        [InlineData(4)]
        [InlineData(5)]
        [InlineData(6)]
        public void GraphSettings_CompactWithoutBuildingTheNetwork(int seed)
        {
            GraphNetworkSettings settings = RandomGraph(seed);
            IGraphNetwork network = NetworkFactory.BuildGraph(settings);
            ICompactNetwork fromNetwork = NetworkCompactor.Compact(network);
            ICompactNetwork fromSettings = NetworkCompactor.CompactGraph(settings);

            Assert.Equal(fromNetwork.ParameterCount, fromSettings.ParameterCount);
            Assert.Equal(fromNetwork.OutputCount, fromSettings.OutputCount);
            AssertSameOutputs(network, fromSettings, Samples(3, 10, seed));
        }

        [Fact]
        public void GraphSettings_AreValidatedLikeBuildGraph()
        {
            var cycle = new GraphNetworkSettings
            {
                Nodes = { new GraphNode(0, NodeType.INPUT), new GraphNode(1, NodeType.HIDDEN), new GraphNode(2, NodeType.HIDDEN), new GraphNode(3, NodeType.OUTPUT) },
                Connections = { new GraphConnection(0, 0, 1, 1), new GraphConnection(1, 1, 2, 1), new GraphConnection(2, 2, 1, 1), new GraphConnection(3, 2, 3, 1) }
            };

            Assert.Throws<ArgumentException>(() => NetworkCompactor.CompactGraph(cycle));
            Assert.Throws<ArgumentNullException>(() => NetworkCompactor.CompactGraph(null));
        }

        [Theory]
        [InlineData(LayerConnectionType.Convolutional)]
        [InlineData(LayerConnectionType.Deconvolutional)]
        public void LayeredConvolution_WithSharedWeights_GivesTheSameOutputs(LayerConnectionType type)
        {
            Network network = LocalConnectionTests.Build(8, type, kernel: 3, stride: 2, padding: 1, filters: 3, inputBias: true,
                                                         hiddenActivation: ActivationTypes.HyperbolicTangens, outputs: 2);

            AssertSameOutputs(network, NetworkCompactor.Compact(network), Samples(8));
        }

        [Fact]
        public void ResNet18_GivesTheInferenceOutputsWithBatchNormalizationFoldedIntoTheConvolutions()
        {
            IConvolutionalNetwork network = ResNetTests.TrainedNarrowResNet18();
            network.Training = false;

            var compact = (CompactConvolutionalNetwork)NetworkCompactor.Compact(network);

            AssertSameOutputs(network, compact, Samples(3 * 32 * 32, 3));

            // 51 layers (counting those inside the 8 residual blocks) become 31 operations: all 20 batch normalizations
            // follow a convolution and are folded into it, leaving one bias per channel instead of gamma and beta
            int normalizedChannels = ((ConvolutionalNetwork)network).Layers
                .SelectMany(x => x is ResidualBlockLayer block ? block.Main.Concat(block.Shortcut) : new[] { x })
                .OfType<BatchNormalizationLayer>()
                .Sum(x => x.InputShape.Channels);
            Assert.Equal(31, compact.OperationCount);
            Assert.Equal(network.ParameterCount - normalizedChannels, compact.ParameterCount);
        }

        [Fact]
        public void ConvolutionalNetwork_KeepsBatchNormalizationsThatCannotBeFolded()
        {
            IConvolutionalNetwork network = NetworkFactory.BuildConvolutional(new ConvolutionalNetworkSettings
            {
                InputChannels = 2,
                InputHeight = 7,
                InputWidth = 6,
                OutputDimension = 3,
                OutputActivationType = ActivationTypes.Sigmoid,
                Seed = 3,
                Layers =
                {
                    new ConvolutionLayerSettings { Filters = 3, KernelSize = 3, Padding = 1, ActivationType = ActivationTypes.RELU },   // activated: no folding
                    new BatchNormalizationLayerSettings { ActivationType = ActivationTypes.HyperbolicTangens },
                    new PoolingLayerSettings { Type = PoolingType.Average, Size = 2, Padding = 1 },
                    new BatchNormalizationLayerSettings(),                                                                            // after pooling
                    new DenseLayerSettings { NeuronsCount = 5, ActivationType = ActivationTypes.SWISH }
                }
            });
            foreach (LearningRateOptimizer optimizer in network.Optimizers) optimizer.LearningRate = 0.01f;
            foreach (Quad[] sample in Samples(2 * 7 * 6, 4, 4))
            {
                network.Propagate(sample);
                network.Backpropagate(0.2f, 0.5f, 0.8f);
            }

            network.Training = false;
            var compact = (CompactConvolutionalNetwork)NetworkCompactor.Compact(network);

            Assert.Equal(6, compact.OperationCount);
            AssertSameOutputs(network, compact, Samples(2 * 7 * 6, 4, 5));
        }

        [Fact]
        public void CompactNetwork_IsASnapshotAndCanBeUsedFromSeveralThreads()
        {
            INetwork network = Layered(normalize: false, softMax: false);
            ICompactNetwork compact = NetworkCompactor.Compact(network);
            Quad[][] samples = Samples(4, 200, 6);
            float[][] before = samples.Select(x => compact.Propagate(x).Select(v => (float)v).ToArray()).ToArray();

            // training the original afterwards does not change the copy
            foreach (LearningRateOptimizer optimizer in ((Network)network).Optimizers) optimizer.LearningRate = 0.1f;
            for (int i = 0; i < 20; i++)
            {
                network.Propagate(samples[i]);
                network.Backpropagate(1f, 0f, -1f);
            }

            var parallel = new float[samples.Length][];
            Parallel.For(0, samples.Length, i => parallel[i] = compact.Propagate(samples[i]).Select(v => (float)v).ToArray());

            for (int i = 0; i < samples.Length; i++)
            {
                Assert.Equal(before[i], parallel[i]);
            }
        }

        [Fact]
        public void Compact_RejectsWhatItCannotHandle()
        {
            Assert.Throws<ArgumentNullException>(() => NetworkCompactor.Compact(null));
            Assert.Throws<ArgumentException>(() => NetworkCompactor.Compact(Layered(false, false)).Propagate(new Quad[3]));
            Assert.Throws<NotSupportedException>(() => NetworkCompactor.Compact(new UnknownNetwork()));
        }

        private sealed class UnknownNetwork : INetwork
        {
            public Quad[] Values => Array.Empty<Quad>();
            public Quad[] Errors => Array.Empty<Quad>();
            public Quad[] InputGradients => Array.Empty<Quad>();
            public IReadOnlyList<Tuple<Quad, Quad>> MinMaxValues => null;
            public Quad Backpropagate(params Quad[] targets) => 0;
            public Quad[] Propagate(params Quad[] values) => values;
            public void Save(string fullDirPath, string fileName) => throw new NotSupportedException();
        }
    }
}
