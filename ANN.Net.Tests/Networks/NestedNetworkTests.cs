using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.IdentityGenerators;
using Xunit;
using Xunit.Abstractions;

namespace ANN.Net.Tests.Networks
{
    /// <summary>
    /// Networks inside the nodes of other networks
    /// </summary>
    public class NestedNetworkTests
    {
        private readonly ITestOutputHelper output;

        public NestedNetworkTests(ITestOutputHelper output) => this.output = output;

        private static IMatrixNetwork Matrix(int inputs, int outputs, int seed, bool softMax = false, bool normalize = false, int hidden = 6,
                                             OptimizerTypes optimizer = OptimizerTypes.Normal, ActivationTypes output = ActivationTypes.HyperbolicTangens)
        {
            return NetworkFactory.BuildMatrixFFN(new NetworkSettings
            {
                Seed = seed,
                LossType = LossFunctionTypes.Squared,
                IdentityGenerator = new IncrementalGenerator(),
                OutputApplySoftMax = softMax,
                ShouldNormalize = normalize,
                MinMaxValues = normalize ? Enumerable.Range(0, Math.Max(inputs, outputs)).Select(i => Tuple.Create<Quad, Quad>(-2 - i, 3 + i)).ToList() : null,
                InputSettings = new InputLayerSettings { NeuronsCount = (ushort)inputs, HasBiasNeuron = true, ActivationType = normalize ? ActivationTypes.SoftSign : ActivationTypes.Identity },
                OutputSettings = new OutputLayerSettings { NeuronsCount = (ushort)outputs, ActivationType = softMax ? ActivationTypes.Sigmoid : output, OptimizerType = optimizer, InitializationType = WeightInitType.Xavier }
            },
            new RecurrentLayerSettings
            {
                NeuronsCount = (ushort)hidden, LayersCount = 1, CellType = NeuronType.Normal, PrevLayerConnectionType = LayerConnectionType.FullyConnected,
                ActivationType = ActivationTypes.HyperbolicTangens, HasBiasNeuron = true, OptimizerType = optimizer, InitializationType = WeightInitType.Xavier
            });
        }

        private static INetwork Neurons(int inputs, int outputs, int seed, bool softMax) => NetworkFactory.BuildFFN(new NetworkSettings
        {
            Seed = seed,
            IdentityGenerator = new IncrementalGenerator(),
            OutputApplySoftMax = softMax,
            InputSettings = new InputLayerSettings { NeuronsCount = (ushort)inputs, HasBiasNeuron = true },
            OutputSettings = new OutputLayerSettings { NeuronsCount = (ushort)outputs, ActivationType = ActivationTypes.Sigmoid, InitializationType = WeightInitType.Xavier }
        },
        new RecurrentLayerSettings
        {
            NeuronsCount = 5, LayersCount = 1, CellType = NeuronType.Normal, PrevLayerConnectionType = LayerConnectionType.FullyConnected,
            ActivationType = ActivationTypes.Sigmoid, HasBiasNeuron = true, InitializationType = WeightInitType.Xavier
        });

        private static IGraphNetwork Graph() => NetworkFactory.BuildGraph(new GraphNetworkSettings
        {
            Nodes =
            {
                new GraphNode(0, NodeType.INPUT), new GraphNode(1, NodeType.INPUT), new GraphNode(2, NodeType.INPUT), new GraphNode(3, NodeType.BIAS),
                new GraphNode(10, NodeType.HIDDEN, ActivationTypes.HyperbolicTangens), new GraphNode(11, NodeType.HIDDEN, ActivationTypes.Sigmoid),
                new GraphNode(20, NodeType.OUTPUT), new GraphNode(21, NodeType.OUTPUT)
            },
            Connections =
            {
                new GraphConnection(1, 0, 10, 0.8f), new GraphConnection(2, 1, 10, -0.5f), new GraphConnection(3, 2, 11, 1.2f), new GraphConnection(4, 3, 11, 0.3f),
                new GraphConnection(5, 10, 20, 0.9f), new GraphConnection(6, 11, 20, -0.7f), new GraphConnection(7, 0, 21, 0.4f), new GraphConnection(8, 10, 21, 1.1f)
            },
            OutputActivationType = ActivationTypes.Sigmoid
        });

        private static IConvolutionalNetwork Convolutional(int seed) => NetworkFactory.BuildConvolutional(new ConvolutionalNetworkSettings
        {
            InputChannels = 2, InputHeight = 5, InputWidth = 5, OutputDimension = 3, OutputApplySoftMax = true, Seed = seed,
            Layers =
            {
                new ConvolutionLayerSettings { Filters = 3, KernelSize = 3, Padding = 1, ActivationType = ActivationTypes.HyperbolicTangens },
                new PoolingLayerSettings { Type = PoolingType.Average, Size = 2 },
                new DenseLayerSettings { NeuronsCount = 4, ActivationType = ActivationTypes.Sigmoid }
            }
        });

        private static IVectorNetwork Transformer(int seed) => NetworkFactory.AsVectorNetwork(NetworkFactory.BuildTransformer(new TransformerSettings
        {
            InputDimension = 3, ModelDimension = 8, HeadsCount = 2, FeedForwardDimension = 12, LayersCount = 1, OutputDimension = 3,
            MaxSequenceLength = 6, UseCausalMask = true, OutputApplySoftMax = true, FeedForwardActivationType = ActivationTypes.HyperbolicTangens, Seed = seed
        }), 4);

        private static void Freeze(IVectorNetwork network)
        {
            foreach (LearningRateOptimizer optimizer in network.Optimizers) optimizer.LearningRate = 0;
        }

        private static Quad[] Vector(int size, int seed, double scale = 1)
        {
            var random = new Random(seed);
            return Enumerable.Range(0, size).Select(_ => (Quad)(((random.NextDouble() * 2) - 1) * scale)).ToArray();
        }

        /// <summary>
        /// The input gradient of L = Σ c_i y_i against central differences
        /// </summary>
        private static void AssertGradient(IVectorNetwork network, Quad[] x, int seed)
        {
            Freeze(network);
            Quad[] c = Vector(network.OutputSize, seed + 100);
            double Loss(Quad[] input) => network.Propagate(input).Zip(c, (y, w) => (double)(float)y * (float)w).Sum();

            network.Propagate(x);
            Quad[] analytic = network.BackpropagateGradient(c);
            Assert.Equal(network.InputSize, analytic.Length);

            const float step = 1e-2f;
            double largest = 0, scale = 0;
            for (int k = 0; k < x.Length; k++)
            {
                Quad[] plus = (Quad[])x.Clone(), minus = (Quad[])x.Clone();
                plus[k] += step;
                minus[k] -= step;
                double numeric = (Loss(plus) - Loss(minus)) / (2 * step);
                largest = Math.Max(largest, Math.Abs(numeric - analytic[k]));
                scale = Math.Max(scale, Math.Abs(numeric));
            }

            Assert.True(largest <= 2e-3 + (2e-2 * scale), $"{network.GetType().Name}: gradients differ by {largest} (largest gradient {scale})");
        }

        [Fact]
        public void EveryVectorNetwork_PassesTheRightGradient()
        {
            AssertGradient(Matrix(4, 3, 1), Vector(4, 1), 1);
            AssertGradient(Matrix(4, 3, 2, softMax: true, normalize: true), Vector(4, 2), 2);
            AssertGradient(NetworkFactory.AsVectorNetwork(Neurons(3, 4, 3, softMax: true)), Vector(3, 3), 3);
            AssertGradient(Graph(), Vector(3, 4), 4);
            AssertGradient(Convolutional(5), Vector(50, 5), 5);
            AssertGradient(Transformer(6), Vector(12, 6), 6);
        }

        /// <summary>
        /// Nodes that read the input twice, feed several nodes, are summed and concatenated, and hold a nested network
        /// </summary>
        private static INestedNetwork Composite(int seed, IVectorNetwork sequence = null)
        {
            INestedNetwork inner = NetworkFactory.BuildNested(new NestedNetworkSettings
            {
                InputSize = 6,
                Nodes =
                {
                    new NestedNode(1, Matrix(6, 4, seed + 10), NestedNetworkSettings.InputNode),
                    new NestedNode(2, Matrix(6, 4, seed + 11), NestedNetworkSettings.InputNode),
                    new NestedNode(3, Matrix(4, 5, seed + 12), 1, 2) { Combination = NodeCombination.Sum }
                },
                Outputs = { 3, 1 }
            });

            var settings = new NestedNetworkSettings
            {
                InputSize = 4,
                Nodes =
                {
                    new NestedNode(1, Matrix(4, 6, seed), NestedNetworkSettings.InputNode),
                    new NestedNode(2, inner, 1),
                    new NestedNode(3, Matrix(13, 12, seed + 1), 2, NestedNetworkSettings.InputNode)   // 9 + 4 values
                },
                Outputs = { 3 }
            };
            if (sequence != null)
            {
                settings.Nodes.Add(new NestedNode(4, sequence, 3));
                settings.Outputs = new List<uint> { 4, 1 };
            }

            return NetworkFactory.BuildNested(settings);
        }

        [Fact]
        public void NestedNetwork_PassesTheRightGradient()
        {
            AssertGradient(Composite(1), Vector(4, 7), 7);
            AssertGradient(Composite(2, Transformer(3)), Vector(4, 8), 8);
        }

        [Fact]
        public void SingleNode_TrainsExactlyLikeTheNetworkInside()
        {
            IMatrixNetwork alone = Matrix(5, 3, 9, optimizer: OptimizerTypes.Adam);
            IMatrixNetwork inside = Matrix(5, 3, 9, optimizer: OptimizerTypes.Adam);
            INestedNetwork nested = NetworkFactory.BuildNested(new NestedNetworkSettings
            {
                InputSize = 5,
                Nodes = { new NestedNode(1, inside, NestedNetworkSettings.InputNode) },
                Outputs = { 1 }
            });
            foreach (LearningRateOptimizer optimizer in alone.Optimizers.Concat(nested.Optimizers)) optimizer.LearningRate = 0.02f;

            for (int step = 0; step < 30; step++)
            {
                Quad[] x = Vector(5, step), target = Vector(3, step + 50, 0.8);
                Assert.Equal(alone.Propagate(x).Select(v => (float)v), nested.Propagate(x).Select(v => (float)v));
                Assert.Equal((float)alone.Backpropagate(target), (float)nested.Backpropagate(target));
            }
        }

        [Fact]
        public void ThreeLevelNetwork_LearnsAFunction()
        {
            INestedNetwork network = Composite(4, Transformer(5));
            foreach (LearningRateOptimizer optimizer in network.Optimizers) optimizer.LearningRate = 0.03f;

            Quad[][] inputs = Enumerable.Range(0, 12).Select(i => Vector(4, 200 + i)).ToArray();
            Quad[][] targets = inputs.Select(x =>
            {
                // per token, a distribution that depends on the inputs; then the 6 values of node 1
                var result = new List<Quad>();
                for (int t = 0; t < 4; t++)
                {
                    float a = MathF.Exp((float)x[t]), b = MathF.Exp((float)x[(t + 1) % 4]), c = 1;
                    result.AddRange(new Quad[] { a / (a + b + c), b / (a + b + c), c / (a + b + c) });
                }

                result.AddRange(Enumerable.Range(0, 6).Select(i => (Quad)(0.5f * MathF.Sin((float)x[i % 4] + i))));
                return result.ToArray();
            }).ToArray();

            double Loss() => inputs.Zip(targets, (x, t) => network.Propagate(x).Zip(t, (y, v) => Math.Pow(y - v, 2)).Sum()).Average();
            double before = Loss();
            for (int epoch = 0; epoch < 300; epoch++)
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    network.Propagate(inputs[i]);
                    network.Backpropagate(targets[i]);
                }
            }

            double after = Loss();
            output.WriteLine($"{network.ParameterCount} parameters in {network.Nodes.Count} nodes; loss {before:G4} -> {after:G4}");
            Assert.True(after < before / 5, $"loss {before} -> {after}");
        }

        [Fact]
        public void SaveAndLoad_KeepsEveryLevel()
        {
            INestedNetwork original = Composite(6);
            foreach (LearningRateOptimizer optimizer in original.Optimizers) optimizer.LearningRate = 0.05f;
            Quad[] x = Vector(4, 9), target = Vector(12, 10, 0.5);
            for (int i = 0; i < 5; i++)
            {
                original.Propagate(x);
                original.Backpropagate(target);
            }

            using var stream = new MemoryStream();
            NetworkSerializer.Save(original, stream);
            INestedNetwork loaded = NetworkSerializer.Load<INestedNetwork>(new MemoryStream(stream.ToArray()));

            Assert.Equal(original.ParameterCount, loaded.ParameterCount);
            Assert.IsAssignableFrom<INestedNetwork>(loaded.Nodes[2]);
            for (int i = 0; i < 5; i++)
            {
                Assert.Equal(original.Propagate(x).Select(v => (float)v), loaded.Propagate(x).Select(v => (float)v));
                Assert.Equal((float)original.Backpropagate(target), (float)loaded.Backpropagate(target));
            }

            Assert.Throws<NotSupportedException>(() => NetworkSerializer.Save(Composite(7, Transformer(8)), new MemoryStream()));
        }

        [Fact]
        public void NodesThatFeedNoOutput_AreLeftOut()
        {
            INestedNetwork network = NetworkFactory.BuildNested(new NestedNetworkSettings
            {
                InputSize = 3,
                Nodes =
                {
                    new NestedNode(1, Matrix(3, 2, 1), NestedNetworkSettings.InputNode),
                    new NestedNode(2, Matrix(3, 2, 2), NestedNetworkSettings.InputNode)
                },
                Outputs = { 1, NestedNetworkSettings.InputNode }
            });

            Assert.Equal(new uint[] { 1 }, network.Nodes.Keys);
            Assert.Equal(5, network.OutputSize);
            Assert.Equal(new Quad[] { 0.5f, -1, 2 }, network.Propagate(0.5f, -1, 2).Skip(2));
        }

        [Fact]
        public void Build_RejectsInvalidNetworks()
        {
            NestedNetworkSettings Settings(params NestedNode[] nodes) => new NestedNetworkSettings { InputSize = 3, Nodes = nodes.ToList(), Outputs = { nodes[^1].Id } };
            IMatrixNetwork shared = Matrix(3, 3, 1);

            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildNested(Settings(new NestedNode(1, Matrix(3, 3, 1), 2), new NestedNode(2, Matrix(3, 3, 2), 1))));
            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildNested(Settings(new NestedNode(1, Matrix(4, 3, 1), NestedNetworkSettings.InputNode))));
            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildNested(Settings(new NestedNode(1, shared, 0), new NestedNode(2, shared, 1))));
            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildNested(Settings(new NestedNode(1, Matrix(3, 3, 1), 7))));
            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildNested(Settings(new NestedNode(0, Matrix(3, 3, 1), 0))));
            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildNested(Settings(new NestedNode(1, Matrix(3, 3, 1), 0), new NestedNode(2, Matrix(4, 3, 2), 0, 1) { Combination = NodeCombination.Sum })));
            Assert.Throws<NotSupportedException>(() => NetworkCompactor.Compact(NetworkFactory.BuildNested(Settings(new NestedNode(1, Matrix(3, 3, 1), 0)))));
        }
    }
}
