using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Settings;
using ANN.Net.IdentityGenerators;
using Microsoft.ML.OnnxRuntime;
using Xunit;

namespace ANN.Net.Tests.Networks
{
    /// <summary>
    /// Networks exported by <see cref="OnnxExporter"/> and run in ONNX Runtime compute the same values as <see cref="INetwork.Propagate"/>
    /// </summary>
    public class OnnxExportTests
    {
        private static float[][] Samples(int width)
        {
            var random = new Random(1);
            return Enumerable.Range(0, 8)
                             .Select(i => Enumerable.Range(0, width).Select(_ => i == 0 ? 0f : (float)((random.NextDouble() * 6) - 3)).ToArray())
                             .ToArray();
        }

        internal static float[][] Run(byte[] model, float[][] samples, string inputName = "input")
        {
            using var session = new InferenceSession(model);
            int width = samples[0].Length;
            float[] data = samples.SelectMany(x => x).ToArray();

            using OrtValue input = OrtValue.CreateTensorValueFromMemory(data, new long[] { samples.Length, width });
            using var runOptions = new RunOptions();
            using var results = session.Run(runOptions, new[] { inputName }, new[] { input }, session.OutputNames);

            float[] output = results[0].GetTensorDataAsSpan<float>().ToArray();
            int outputs = output.Length / samples.Length;
            return Enumerable.Range(0, samples.Length).Select(i => output.Skip(i * outputs).Take(outputs).ToArray()).ToArray();
        }

        private static void AssertSameAsNetwork(INetwork network, int inputs, OnnxExportOptions options = null)
        {
            float[][] samples = Samples(inputs);
            float[][] actual = Run(OnnxExporter.Export(network, options), samples, options?.InputName ?? "input");

            for (int i = 0; i < samples.Length; i++)
            {
                float[] expected = network.Propagate(samples[i].Select(x => (Quad)x).ToArray()).Select(x => (float)x).ToArray();
                Assert.Equal(expected.Length, actual[i].Length);
                for (int j = 0; j < expected.Length; j++)
                {
                    Assert.Equal(expected[j], actual[i][j], 5);
                }
            }
        }

        private static INetwork Layered(bool normalize = false, bool softMax = false, ActivationTypes inputActivation = ActivationTypes.Identity)
        {
            return NetworkFactory.BuildFFN(new NetworkSettings
            {
                IdentityGenerator = new IncrementalGenerator(),
                LossType = LossFunctionTypes.Squared,
                ShouldNormalize = normalize,
                MinMaxValues = normalize
                    ? new List<Tuple<Quad, Quad>> { Tuple.Create<Quad, Quad>(-5, 5), Tuple.Create<Quad, Quad>(-4, 6), Tuple.Create<Quad, Quad>(0, 2) }
                    : null,
                OutputApplySoftMax = softMax,
                InputSettings = new InputLayerSettings { NeuronsCount = 3, ActivationType = inputActivation, HasBiasNeuron = true },
                OutputSettings = new OutputLayerSettings { NeuronsCount = 2, ActivationType = ActivationTypes.Sigmoid }
            },
            new RecurrentLayerSettings
            {
                NeuronsCount = 5,
                LayersCount = 2,
                CellType = NeuronType.Normal,
                PrevLayerConnectionType = LayerConnectionType.FullyConnected,
                ActivationType = ActivationTypes.HyperbolicTangens,
                HasBiasNeuron = true
            });
        }

        [Fact]
        public void Layered_MatchesPropagate()
        {
            AssertSameAsNetwork(Layered(), 3);
        }

        [Fact]
        public void Layered_WithNormalizationSoftMaxAndActivatedInputs_MatchesPropagate()
        {
            AssertSameAsNetwork(Layered(normalize: true, softMax: true, inputActivation: ActivationTypes.Sigmoid), 3);
        }

        [Fact]
        public void Graph_WithEveryActivationAndSkipConnections_MatchesPropagate()
        {
            IGraphNetwork network = NetworkFactory.BuildGraph(new GraphNetworkSettings
            {
                Nodes =
                {
                    new GraphNode(0, NodeType.INPUT), new GraphNode(1, NodeType.INPUT), new GraphNode(2, NodeType.INPUT, ActivationTypes.ArcusTangens),
                    new GraphNode(3, NodeType.BIAS),
                    new GraphNode(10, NodeType.HIDDEN, ActivationTypes.RELU), new GraphNode(11, NodeType.HIDDEN, ActivationTypes.LeakyRELU),
                    new GraphNode(12, NodeType.HIDDEN, ActivationTypes.BinaryRELU), new GraphNode(13, NodeType.HIDDEN),
                    new GraphNode(14, NodeType.HIDDEN, ActivationTypes.Identity), // fed only by the bias
                    new GraphNode(20, NodeType.OUTPUT), new GraphNode(21, NodeType.OUTPUT, ActivationTypes.ArcusTangens),
                    new GraphNode(22, NodeType.OUTPUT) // not connected: stays 0
                },
                Connections =
                {
                    new GraphConnection(1, 0, 10, 1.5f), new GraphConnection(2, 1, 11, -2f), new GraphConnection(3, 2, 12, 0.7f),
                    new GraphConnection(4, 10, 13, 0.3f), new GraphConnection(5, 11, 13, -0.4f), new GraphConnection(6, 13, 20, 1.1f),
                    new GraphConnection(7, 0, 20, -0.6f), new GraphConnection(8, 12, 21, 2f), new GraphConnection(9, 3, 21, -1f),
                    new GraphConnection(10, 3, 14, 0.5f), new GraphConnection(11, 14, 20, 0.9f), new GraphConnection(12, 1, 21, 0.25f),
                    new GraphConnection(13, 3, 13, 0.2f)
                }
            });

            AssertSameAsNetwork(network, 3);
        }

        [Fact]
        public void Export_ToFileWithCustomNames()
        {
            INetwork network = Layered();
            var options = new OnnxExportOptions { InputName = "features", OutputName = "scores", GraphName = "classifier" };
            string file = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".onnx");

            try
            {
                OnnxExporter.Export(network, file, options);

                using var session = new InferenceSession(file);
                Assert.Equal(new[] { "features" }, session.InputNames);
                Assert.Equal(new[] { "scores" }, session.OutputNames);
                Assert.Equal(new[] { -1, 3 }, session.InputMetadata["features"].Dimensions);
                Assert.Equal(new[] { -1, 2 }, session.OutputMetadata["scores"].Dimensions);
            }
            finally
            {
                File.Delete(file);
            }

            AssertSameAsNetwork(network, 3, options);
        }

        [Fact]
        public void Export_RejectsNetworksItDoesNotKnow()
        {
            Assert.Throws<NotSupportedException>(() => OnnxExporter.Export(new UnknownNetwork()));
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
