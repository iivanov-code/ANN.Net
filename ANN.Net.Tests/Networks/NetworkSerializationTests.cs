using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.IdentityGenerators;
using Xunit;

namespace ANN.Net.Tests.Networks
{
    /// <summary>
    /// Saving networks to JSON with <see cref="NetworkSerializer"/> and loading them back
    /// </summary>
    public class NetworkSerializationTests
    {
        private static readonly Quad[][] Samples =
        {
            new Quad[] { 0.1f, 0.9f },
            new Quad[] { -1.5f, 0.25f },
            new Quad[] { 2f, -3f },
            new Quad[] { 0f, 0f }
        };

        private static INetwork BuildLayered(OptimizerTypes optimizer = OptimizerTypes.Normal, bool normalize = false, bool softMax = false)
        {
            INetwork network = NetworkFactory.BuildFFN(new NetworkSettings
            {
                IdentityGenerator = new IncrementalGenerator(),
                LossType = LossFunctionTypes.Squared,
                ShouldNormalize = normalize,
                MinMaxValues = normalize ? new List<Tuple<Quad, Quad>> { Tuple.Create<Quad, Quad>(-5, 5), Tuple.Create<Quad, Quad>(-4, 6) } : null,
                OutputApplySoftMax = softMax,
                InputSettings = new InputLayerSettings { NeuronsCount = 2, ActivationType = ActivationTypes.Identity, HasBiasNeuron = true, OptimizerType = optimizer },
                OutputSettings = new OutputLayerSettings { NeuronsCount = 2, ActivationType = ActivationTypes.Sigmoid, OptimizerType = optimizer }
            },
            new RecurrentLayerSettings
            {
                NeuronsCount = 4,
                LayersCount = 2,
                CellType = NeuronType.Normal,
                PrevLayerConnectionType = LayerConnectionType.FullyConnected,
                ActivationType = ActivationTypes.HyperbolicTangens,
                HasBiasNeuron = true,
                OptimizerType = optimizer
            });

            SetLearningRate(network, 0.1f);
            return network;
        }

        private static IGraphNetwork BuildGraph()
        {
            IGraphNetwork network = NetworkFactory.BuildGraph(new GraphNetworkSettings
            {
                Nodes =
                {
                    new GraphNode(10, NodeType.INPUT), new GraphNode(11, NodeType.INPUT), new GraphNode(12, NodeType.BIAS),
                    new GraphNode(20, NodeType.HIDDEN), new GraphNode(21, NodeType.HIDDEN, ActivationTypes.RELU),
                    new GraphNode(22, NodeType.HIDDEN), // dead end, pruned
                    new GraphNode(30, NodeType.OUTPUT)
                },
                Connections =
                {
                    new GraphConnection(1, 10, 20, 0.5f), new GraphConnection(2, 11, 20, -1.25f), new GraphConnection(3, 20, 21, 2f),
                    new GraphConnection(4, 21, 30, 0.75f), new GraphConnection(5, 10, 30, -0.3f), new GraphConnection(6, 12, 30, 0.1f),
                    new GraphConnection(7, 12, 21, 0.2f), new GraphConnection(8, 10, 22, 1f), new GraphConnection(9, 11, 30, 3f, enabled: false)
                },
                HiddenActivationType = ActivationTypes.Sigmoid,
                OutputActivationType = ActivationTypes.HyperbolicTangens
            });

            SetLearningRate(network, 0.05f);
            return network;
        }

        private static void SetLearningRate(INetwork network, float learningRate)
        {
            IEnumerable<LearningRateOptimizer> optimizers = network is IGraphNetwork graph ? graph.Optimizers : ((Network)network).Optimizers;
            foreach (LearningRateOptimizer optimizer in optimizers)
            {
                optimizer.LearningRate = learningRate;
            }
        }

        private static T RoundTrip<T>(T network)
            where T : INetwork
        {
            using var stream = new MemoryStream();
            NetworkSerializer.Save(network, stream);
            stream.Position = 0;
            return NetworkSerializer.Load<T>(stream);
        }

        private static void AssertSameOutputs(INetwork expected, INetwork actual)
        {
            foreach (Quad[] sample in Samples)
            {
                Assert.Equal(expected.Propagate(sample).Select(x => (float)x), actual.Propagate(sample).Select(x => (float)x));
            }
        }

        private static void Train(INetwork network, int steps)
        {
            for (int i = 0; i < steps; i++)
            {
                Quad[] sample = Samples[i % Samples.Length];
                int outputs = network.Propagate(sample).Length;
                network.Backpropagate(new Quad[] { 0.2f, 0.7f }.Take(outputs).ToArray());
            }
        }

        [Fact]
        public void Layered_RoundTrip_GivesTheSameOutputsAndKeepsTraining()
        {
            INetwork original = BuildLayered();
            Train(original, 20);

            INetwork loaded = RoundTrip(original);

            AssertSameOutputs(original, loaded);

            // same weights, learning rates and synapse order: training on goes exactly the same way
            Train(original, 20);
            Train(loaded, 20);
            AssertSameOutputs(original, loaded);
        }

        [Fact]
        public void Layered_RoundTrip_KeepsNormalizationAndSoftMax()
        {
            INetwork original = BuildLayered(OptimizerTypes.Adam, normalize: true, softMax: true);

            INetwork loaded = RoundTrip(original);

            AssertSameOutputs(original, loaded);
            Assert.Equal(original.MinMaxValues.Select(x => ((float)x.Item1, (float)x.Item2)), loaded.MinMaxValues.Select(x => ((float)x.Item1, (float)x.Item2)));
            Assert.All(((Network)loaded).Optimizers, x => Assert.Equal(0.1f, (float)x.LearningRate));
        }

        [Fact]
        public void Graph_SaveToFile_LoadsAGraphNetworkWithTheSameConnections()
        {
            IGraphNetwork original = BuildGraph();
            Train(original, 10);
            string directory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(directory);

            try
            {
                original.Save(directory, "graph");
                IGraphNetwork loaded = NetworkSerializer.Load<IGraphNetwork>(Path.Combine(directory, "graph.ann"));

                AssertSameOutputs(original, loaded);
                Assert.Equal(original.ConnectionWeights.OrderBy(x => x.Key).Select(x => (x.Key, (float)x.Value)),
                             loaded.ConnectionWeights.OrderBy(x => x.Key).Select(x => (x.Key, (float)x.Value)));
                Assert.Equal(new uint[] { 20, 21 }, loaded.HiddenNodeIds.OrderBy(x => x));

                Train(original, 10);
                Train(loaded, 10);
                AssertSameOutputs(original, loaded);
            }
            finally
            {
                Directory.Delete(directory, recursive: true);
            }
        }

        [Fact]
        public void Load_ThroughANNetwork_ReturnsTheSavedNetwork()
        {
            INetwork original = BuildLayered();
            string file = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".ann");

            try
            {
                NetworkSerializer.Save(original, file);
                AssertSameOutputs(original, ANNetwork.Load(file));
            }
            finally
            {
                File.Delete(file);
            }
        }

        [Fact]
        public void Load_RejectsFilesThatAreNotNetworks()
        {
            static INetwork LoadText(string text) => NetworkSerializer.Load(new MemoryStream(Encoding.UTF8.GetBytes(text)));

            Assert.Throws<InvalidDataException>(() => LoadText("not json"));
            Assert.Throws<InvalidDataException>(() => LoadText("{ \"format\": \"something else\", \"version\": 1 }"));
            Assert.Throws<InvalidDataException>(() => LoadText("{ \"format\": \"ANN.Net network\", \"version\": 99 }"));

            using var stream = new MemoryStream();
            NetworkSerializer.Save(BuildLayered(), stream);
            string json = Encoding.UTF8.GetString(stream.ToArray());

            Assert.Throws<InvalidDataException>(() => LoadText(json.Replace("\"to\": ", "\"to\": 1000", StringComparison.Ordinal)));
            Assert.Throws<InvalidDataException>(() => NetworkSerializer.Load<IGraphNetwork>(new MemoryStream(stream.ToArray())));
        }

        [Fact]
        public void Save_RejectsNetworksItDoesNotKnow()
        {
            Assert.Throws<NotSupportedException>(() => NetworkSerializer.Save(new UnknownNetwork(), new MemoryStream()));
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
