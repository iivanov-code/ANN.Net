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
    /// Every activation a network can be built with: values, derivatives, saving and ONNX export
    /// </summary>
    public class ActivationFunctionTests
    {
        public static readonly ActivationTypes[] Buildable =
        {
            ActivationTypes.Identity, ActivationTypes.Sigmoid, ActivationTypes.HyperbolicTangens, ActivationTypes.ArcusTangens,
            ActivationTypes.RELU, ActivationTypes.LeakyRELU, ActivationTypes.BinaryRELU, ActivationTypes.Gaussian,
            ActivationTypes.Sinusoid, ActivationTypes.SoftSign, ActivationTypes.SQNL, ActivationTypes.ISRU,
            ActivationTypes.GELU, ActivationTypes.SWISH
        };

        public static TheoryData<ActivationTypes> BuildableData()
        {
            var data = new TheoryData<ActivationTypes>();
            foreach (ActivationTypes type in Buildable) data.Add(type);
            return data;
        }

        private static float Activate(ActivationTypes type, float x)
        {
            Quad value = x;
            return NetworkUtils.GetActivation(type).Activation(ref value);
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
        public void Activation_GivesTheKnownValue(ActivationTypes type, float x, float expected)
        {
            Assert.Equal(expected, Activate(type, x), 5);
        }

        [Theory]
        [MemberData(nameof(BuildableData))]
        public void Derivative_MatchesTheFiniteDifference(ActivationTypes type)
        {
            IActivationFunction function = NetworkUtils.GetActivation(type);
            const double h = 1e-3;

            // points away from the kinks of ReLU, the step and SQNL (0 and ±2)
            foreach (float x in new[] { -3.3f, -1.7f, -0.6f, 0.4f, 1.3f, 2.6f })
            {
                double numeric = (Activate(type, (float)(x + h)) - Activate(type, (float)(x - h))) / (2 * h);
                Assert.True(Math.Abs(numeric - (float)function.Derivative(x)) < 1e-2,
                            $"{type} at {x}: derivative {(float)function.Derivative(x)}, finite difference {numeric}");
            }
        }

        private static IGraphNetwork OneNodePerActivation()
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

            return NetworkFactory.BuildGraph(settings);
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
        }

        [Fact]
        public void EveryActivation_ExportsToOnnx()
        {
            IGraphNetwork network = OneNodePerActivation();
            float[][] samples = new[] { -2.5f, -1.1f, -0.3f, 0f, 0.7f, 1.9f, 3f }.Select(x => new[] { x }).ToArray();

            float[][] actual = OnnxExportTests.Run(OnnxExporter.Export(network), samples);

            for (int i = 0; i < samples.Length; i++)
            {
                Assert.Equal((float)network.Propagate(samples[i][0])[0], actual[i][0], 4);
            }
        }
    }
}
