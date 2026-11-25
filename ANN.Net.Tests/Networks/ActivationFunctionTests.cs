using System;
using System.IO;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Utils;
using Xunit;

namespace ANN.Net.Tests.Networks
{
    /// <summary>
    /// Every activation a network can be built with: values, derivatives, saving, compacting and ONNX export. Every
    /// <see cref="ActivationTypes"/> value is either <see cref="Buildable"/> or <see cref="Rejected"/>
    /// </summary>
    public class ActivationFunctionTests
    {
        public static readonly ActivationTypes[] Buildable =
        {
            ActivationTypes.Identity, ActivationTypes.Sigmoid, ActivationTypes.SoftStep, ActivationTypes.SoftPlus,
            ActivationTypes.HyperbolicTangens, ActivationTypes.ArcusTangens, ActivationTypes.RELU, ActivationTypes.LeakyRELU,
            ActivationTypes.ParametricRELU, ActivationTypes.BinaryRELU, ActivationTypes.BinaryStep, ActivationTypes.Gaussian,
            ActivationTypes.Sinusoid, ActivationTypes.SoftSign, ActivationTypes.SQNL, ActivationTypes.ISRU,
            ActivationTypes.GELU, ActivationTypes.SWISH, ActivationTypes.HardSigmoid, ActivationTypes.TanHShrink,
            ActivationTypes.SoftShrink, ActivationTypes.HardShrink
        };

        /// <summary>
        /// Not functions of one neuron's sum: soft-max needs the whole layer (OutputApplySoftMax), BReLU the neuron's position
        /// </summary>
        public static readonly ActivationTypes[] Rejected = { ActivationTypes.SoftMax, ActivationTypes.BReLU };

        public static TheoryData<ActivationTypes> BuildableData()
        {
            var data = new TheoryData<ActivationTypes>();
            foreach (ActivationTypes type in Buildable) data.Add(type);
            return data;
        }

        public static TheoryData<ActivationTypes> RejectedData()
        {
            var data = new TheoryData<ActivationTypes>();
            foreach (ActivationTypes type in Rejected) data.Add(type);
            return data;
        }

        private static float Activate(ActivationTypes type, float x)
        {
            Quad value = x;
            return NetworkUtils.GetActivation(type).Activation(ref value);
        }

        [Fact]
        public void EveryActivationType_IsEitherBuildableOrRejected()
        {
            Assert.Empty(Buildable.Intersect(Rejected));
            Assert.Equal(Buildable.Length, Buildable.Distinct().Count());
            Assert.Equal(Enum.GetValues<ActivationTypes>().OrderBy(x => x), Buildable.Concat(Rejected).OrderBy(x => x));
        }

        [Theory]
        [InlineData(ActivationTypes.Gaussian, 0f, 1f)]
        [InlineData(ActivationTypes.Gaussian, 1f, 0.36787944f)]
        [InlineData(ActivationTypes.Gaussian, -2f, 0.01831564f)]
        [InlineData(ActivationTypes.Sinusoid, 1f, 0.84147098f)]
        [InlineData(ActivationTypes.SoftSign, 1f, 0.5f)]
        [InlineData(ActivationTypes.SoftSign, -3f, -0.75f)]
        [InlineData(ActivationTypes.SQNL, 1f, 0.75f)]
        [InlineData(ActivationTypes.SQNL, -1f, -0.75f)]
        [InlineData(ActivationTypes.SQNL, 3f, 1f)]
        [InlineData(ActivationTypes.SQNL, -3f, -1f)]
        [InlineData(ActivationTypes.ISRU, 1f, 0.70710678f)]
        [InlineData(ActivationTypes.GELU, 1f, 0.84119199f)]
        [InlineData(ActivationTypes.GELU, -1f, -0.15880801f)]
        [InlineData(ActivationTypes.SWISH, 1f, 0.73105858f)]
        [InlineData(ActivationTypes.SWISH, -2f, -0.23840584f)]
        [InlineData(ActivationTypes.Sigmoid, 1f, 0.73105858f)]
        [InlineData(ActivationTypes.SoftStep, 0f, 0.5f)]
        [InlineData(ActivationTypes.SoftStep, 1f, 0.73105858f)]
        [InlineData(ActivationTypes.SoftPlus, 0f, 0.69314718f)]
        [InlineData(ActivationTypes.SoftPlus, 1f, 1.31326169f)]
        [InlineData(ActivationTypes.SoftPlus, -2f, 0.12692801f)]
        [InlineData(ActivationTypes.SoftPlus, 100f, 100f)]
        [InlineData(ActivationTypes.SoftPlus, -100f, 0f)]
        [InlineData(ActivationTypes.ParametricRELU, -2f, -0.5f)]
        [InlineData(ActivationTypes.ParametricRELU, 3f, 3f)]
        [InlineData(ActivationTypes.BinaryStep, -0.1f, 0f)]
        [InlineData(ActivationTypes.BinaryStep, 0f, 1f)]
        [InlineData(ActivationTypes.BinaryStep, 2f, 1f)]
        [InlineData(ActivationTypes.BinaryRELU, -0.1f, 0f)]
        [InlineData(ActivationTypes.BinaryRELU, 0f, 1f)]
        [InlineData(ActivationTypes.HardSigmoid, 0f, 0.5f)]
        [InlineData(ActivationTypes.HardSigmoid, 1.5f, 0.75f)]
        [InlineData(ActivationTypes.HardSigmoid, -1.5f, 0.25f)]
        [InlineData(ActivationTypes.HardSigmoid, 4f, 1f)]
        [InlineData(ActivationTypes.HardSigmoid, -4f, 0f)]
        [InlineData(ActivationTypes.TanHShrink, 1f, 0.23840584f)]
        [InlineData(ActivationTypes.TanHShrink, -2f, -1.03597242f)]
        [InlineData(ActivationTypes.SoftShrink, 0.3f, 0f)]
        [InlineData(ActivationTypes.SoftShrink, -0.5f, 0f)]
        [InlineData(ActivationTypes.SoftShrink, 1.2f, 0.7f)]
        [InlineData(ActivationTypes.SoftShrink, -2f, -1.5f)]
        [InlineData(ActivationTypes.HardShrink, 0.3f, 0f)]
        [InlineData(ActivationTypes.HardShrink, 0.5f, 0f)]
        [InlineData(ActivationTypes.HardShrink, 0.7f, 0.7f)]
        [InlineData(ActivationTypes.HardShrink, -2f, -2f)]
        public void Activation_GivesTheKnownValue(ActivationTypes type, float x, float expected)
        {
            Assert.Equal(expected, Activate(type, x), 5);
        }

        [Theory]
        [MemberData(nameof(BuildableData))]
        public void Activation_DoesNotDependOnEarlierCalls(ActivationTypes type)
        {
            // the functions are shared by every neuron, so they must not keep state between calls
            float[] points = { -2.2f, -0.7f, 0f, 0.9f, 1.8f };
            float[] first = points.Select(x => Activate(type, x)).ToArray();
            float[] second = points.Select(x => Activate(type, x)).ToArray();

            Assert.Equal(first, second);
        }

        [Theory]
        [MemberData(nameof(BuildableData))]
        public void Derivative_MatchesTheFiniteDifference(ActivationTypes type)
        {
            IActivationFunction function = NetworkUtils.GetActivation(type);
            const double h = 1e-3;

            // points away from the kinks of ReLU, the step and SQNL (0 and ±2), the shrinks (±0.5) and the hard sigmoid (±3)
            foreach (float x in new[] { -3.3f, -2.4f, -1.7f, -0.6f, -0.2f, 0.4f, 0.8f, 1.3f, 2.6f, 4.1f })
            {
                double numeric = (Activate(type, (float)(x + h)) - Activate(type, (float)(x - h))) / (2 * h);
                Assert.True(Math.Abs(numeric - (float)function.Derivative(x)) < 1e-2,
                            $"{type} at {x}: derivative {(float)function.Derivative(x)}, finite difference {numeric}");
            }
        }

        [Fact]
        public void StepDerivative_IsZeroAtTheStep()
        {
            Assert.Equal(0f, (float)NetworkUtils.GetActivation(ActivationTypes.BinaryStep).Derivative(0));
            Assert.Equal(0f, (float)NetworkUtils.GetActivation(ActivationTypes.BinaryRELU).Derivative(0));
        }

        [Theory]
        [MemberData(nameof(RejectedData))]
        public void RejectedActivation_ThrowsNotSupported(ActivationTypes type)
        {
            NotSupportedException exception = Assert.Throws<NotSupportedException>(() => NetworkUtils.GetActivation(type));
            Assert.Contains(type.ToString(), exception.Message, StringComparison.Ordinal);
        }

        [Fact]
        public void SoftMax_PointsToOutputApplySoftMax()
        {
            NotSupportedException exception = Assert.Throws<NotSupportedException>(() => NetworkUtils.GetActivation(ActivationTypes.SoftMax));
            Assert.Contains(nameof(NetworkSettings.OutputApplySoftMax), exception.Message, StringComparison.Ordinal);
        }

        [Fact]
        public void UndefinedActivationValue_ThrowsArgumentException()
        {
            Assert.Throws<ArgumentException>(() => NetworkUtils.GetActivation((ActivationTypes)999));
        }

        [Theory]
        [MemberData(nameof(RejectedData))]
        public void RejectedActivation_CannotBuildAnyNetwork(ActivationTypes type)
        {
            Assert.Throws<NotSupportedException>(() => Matrix(type));
            Assert.Throws<NotSupportedException>(() => Matrix(ActivationTypes.RELU, output: type));
            Assert.Throws<NotSupportedException>(() => NetworkFactory.BuildGraph(SingleHidden(type)));
            Assert.Throws<NotSupportedException>(() => NetworkCompactor.CompactGraph(SingleHidden(type)));
        }

        private static IMatrixNetwork Matrix(ActivationTypes hidden, ActivationTypes? output = null)
        {
            return NetworkFactory.BuildMatrixFFN(new NetworkSettings
            {
                Seed = 7,
                InputSettings = new InputLayerSettings { NeuronsCount = 2, ActivationType = ActivationTypes.Identity, HasBiasNeuron = true },
                OutputSettings = new OutputLayerSettings { NeuronsCount = 2, ActivationType = output ?? hidden, InitializationType = WeightInitType.Xavier }
            }, new RecurrentLayerSettings
            {
                NeuronsCount = 4, LayersCount = 1, ActivationType = hidden, PrevLayerConnectionType = LayerConnectionType.FullyConnected,
                HasBiasNeuron = true, InitializationType = WeightInitType.Xavier
            });
        }

        private static GraphNetworkSettings SingleHidden(ActivationTypes type)
        {
            var settings = new GraphNetworkSettings
            {
                Nodes = { new GraphNode(0, NodeType.INPUT), new GraphNode(1, NodeType.HIDDEN, type), new GraphNode(2, NodeType.OUTPUT, ActivationTypes.Identity) },
                OutputActivationType = ActivationTypes.Identity
            };
            settings.Connections.Add(new GraphConnection(0, 0, 1, 0.5f));
            settings.Connections.Add(new GraphConnection(1, 1, 2, 1f));
            return settings;
        }

        /// <summary>
        /// A matrix network with the activation in its hidden and output layers builds, propagates, saves, compacts and
        /// exports to ONNX with the same outputs
        /// </summary>
        [Theory]
        [MemberData(nameof(BuildableData))]
        public void BuildableActivation_WorksInEveryMatrixNetworkForm(ActivationTypes type)
        {
            IMatrixNetwork network = Matrix(type);
            float[][] samples = { new[] { -1.4f, 0.6f }, new[] { 0.2f, -0.3f }, new[] { 2.1f, 1.7f }, new[] { -0.8f, -2.5f } };
            float[][] expected = samples.Select(x => network.Propagate(x.Select(v => (Quad)v).ToArray()).Select(v => (float)v).ToArray()).ToArray();

            using var stream = new MemoryStream();
            NetworkSerializer.Save(network, stream);
            stream.Position = 0;
            INetwork loaded = NetworkSerializer.Load<INetwork>(stream);

            ICompactNetwork compact = NetworkCompactor.Compact(network);
            float[][] onnx = OnnxExportTests.Run(OnnxExporter.Export(network), samples);

            for (int i = 0; i < samples.Length; i++)
            {
                Quad[] input = samples[i].Select(v => (Quad)v).ToArray();
                Assert.Equal(expected[i], loaded.Propagate(input).Select(v => (float)v).ToArray());

                float[] compacted = compact.Propagate(input).Select(v => (float)v).ToArray();
                for (int o = 0; o < expected[i].Length; o++)
                {
                    Assert.Equal(expected[i][o], compacted[o], 4);
                    Assert.Equal(expected[i][o], onnx[i][o], 4);
                }
            }
        }

        private static IGraphNetwork OneNodePerActivation()
        {
            return NetworkFactory.BuildGraph(OneNodePerActivationSettings());
        }

        private static GraphNetworkSettings OneNodePerActivationSettings()
        {
            var settings = new GraphNetworkSettings
            {
                Nodes = { new GraphNode(0, NodeType.INPUT), new GraphNode(1, NodeType.BIAS), new GraphNode(1000, NodeType.OUTPUT, ActivationTypes.Identity) },
                OutputActivationType = ActivationTypes.Identity
            };

            uint connection = 0;
            for (int i = 0; i < Buildable.Length; i++)
            {
                uint hidden = (uint)(10 + i);
                settings.Nodes.Add(new GraphNode(hidden, NodeType.HIDDEN, Buildable[i]));
                settings.Connections.Add(new GraphConnection(connection++, 0, hidden, 1.25f - (0.1f * i)));
                settings.Connections.Add(new GraphConnection(connection++, 1, hidden, 0.05f * i));
                settings.Connections.Add(new GraphConnection(connection++, hidden, 1000, 1f + (0.5f * i)));
            }

            return settings;
        }

        [Fact]
        public void EveryActivation_SurvivesSavingAndLoading()
        {
            IGraphNetwork network = OneNodePerActivation();

            using var stream = new MemoryStream();
            NetworkSerializer.Save(network, stream);
            stream.Position = 0;
            IGraphNetwork loaded = NetworkSerializer.Load<IGraphNetwork>(stream);

            foreach (float x in new[] { -2.5f, -0.3f, 0f, 0.7f, 3f })
            {
                Assert.Equal((float)network.Propagate(x)[0], (float)loaded.Propagate(x)[0]);
            }

            // types that share a function (SoftStep and Sigmoid, BinaryStep and BinaryRELU) keep their own name
            ActivationTypes[] saved = NetworkSerializer.ToDocument(loaded).Neurons.Where(x => x.Type == NodeType.HIDDEN).Select(x => x.Activation).ToArray();
            Assert.Equal(Buildable.OrderBy(x => x), saved.OrderBy(x => x));
        }

        [Fact]
        public void EveryActivation_Compacts()
        {
            IGraphNetwork network = OneNodePerActivation();
            ICompactNetwork compact = NetworkCompactor.CompactGraph(OneNodePerActivationSettings());

            foreach (float x in new[] { -2.5f, -1.1f, -0.3f, 0f, 0.7f, 1.9f, 3f })
            {
                float expected = network.Propagate(x)[0];
                Assert.Equal(expected, (float)compact.Propagate(x)[0], Tolerance(expected));
            }
        }

        [Fact]
        public void EveryActivation_ExportsToOnnx()
        {
            IGraphNetwork network = OneNodePerActivation();
            float[][] samples = new[] { -2.5f, -1.1f, -0.3f, 0f, 0.7f, 1.9f, 3f }.Select(x => new[] { x }).ToArray();

            float[][] actual = OnnxExportTests.Run(OnnxExporter.Export(network), samples);

            for (int i = 0; i < samples.Length; i++)
            {
                float expected = network.Propagate(samples[i][0])[0];
                Assert.Equal(expected, actual[i][0], Tolerance(expected));
            }
        }

        /// <summary>
        /// The graph sums 22 activations with weights up to 11.5, so float rounding grows with the size of the output
        /// </summary>
        private static double Tolerance(float expected) => 1e-5 * Math.Max(1, Math.Abs(expected));
    }
}
