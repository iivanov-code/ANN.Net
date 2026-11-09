using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Abstractions.Spiking;
using ANN.Net.IdentityGenerators;
using ANN.Net.Spiking;
using Xunit;

namespace ANN.Net.Tests.Spiking
{
    /// <summary>
    /// Networks trained with ReLU, leaky ReLU, the identity, sigmoid or tanh compute about the same once converted to
    /// spiking networks, and closer the more time steps they run
    /// </summary>
    public class SpikingConversionTests
    {
        private static NetworkSettings Settings(bool normalize = false, bool softMax = false, ActivationTypes input = ActivationTypes.Identity,
                                                ActivationTypes output = ActivationTypes.Sigmoid) => new NetworkSettings
        {
            Seed = 3,
            IdentityGenerator = new IncrementalGenerator(),
            ShouldNormalize = normalize,
            MinMaxValues = normalize ? Enumerable.Range(0, 5).Select(i => Tuple.Create<Quad, Quad>(-2 - i, 2 + i)).ToList() : null,
            OutputApplySoftMax = softMax,
            InputSettings = new InputLayerSettings { NeuronsCount = 5, HasBiasNeuron = true, ActivationType = input },
            OutputSettings = new OutputLayerSettings { NeuronsCount = 3, ActivationType = output, InitializationType = WeightInitType.Xavier }
        };

        private static RecurrentLayerSettings Hidden(ActivationTypes activation) => new RecurrentLayerSettings
        {
            NeuronsCount = 12,
            LayersCount = 2,
            CellType = NeuronType.Normal,
            PrevLayerConnectionType = LayerConnectionType.FullyConnected,
            ActivationType = activation,
            HasBiasNeuron = true,
            InitializationType = WeightInitType.Xavier
        };

        private static List<float[]> Samples(int count, int seed, int width = 5)
        {
            var random = new Random(seed);
            return Enumerable.Range(0, count).Select(_ => Enumerable.Range(0, width).Select(_ => (float)((random.NextDouble() * 4) - 2)).ToArray()).ToList();
        }

        private static float[] Analog(INetwork network, float[] input) => network.Propagate(input.Select(x => (Quad)x).ToArray()).Select(x => (float)x).ToArray();

        /// <summary>
        /// Largest difference between the spiking and the analog outputs over the samples
        /// </summary>
        private static float Error(INetwork network, ISpikingNetwork spiking, IEnumerable<float[]> samples, int steps)
        {
            float error = 0;
            foreach (float[] input in samples)
            {
                float[] expected = Analog(network, input);
                IReadOnlyList<float> actual = spiking.Run(input, steps).Outputs;
                for (int i = 0; i < expected.Length; i++) error = Math.Max(error, Math.Abs(expected[i] - actual[i]));
            }

            return error;
        }

        private static SpikingConversionOptions Options(int steps = 256) => new SpikingConversionOptions
        {
            CalibrationInputs = Samples(300, 1),
            Percentile = 100,
            TimeSteps = steps
        };

        [Theory]
        [InlineData(ActivationTypes.RELU)]
        [InlineData(ActivationTypes.LeakyRELU)]
        [InlineData(ActivationTypes.Identity)]
        [InlineData(ActivationTypes.Sigmoid)]
        [InlineData(ActivationTypes.HyperbolicTangens)]
        public void ConvertedNetworkComputesWhatTheNetworkComputes(ActivationTypes activation)
        {
            IMatrixNetwork network = NetworkFactory.BuildMatrixFFN(Settings(), Hidden(activation));
            ISpikingNetwork spiking = SpikingNetworkConverter.Convert(network, Options());
            List<float[]> samples = Samples(40, 2);

            float coarse = Error(network, spiking, samples, 16), fine = Error(network, spiking, samples, 512);

            Assert.True(fine < 0.02f, $"error after 512 steps {fine}");
            Assert.True(fine < coarse, $"error after 16 steps {coarse}, after 512 steps {fine}");
            Assert.Equal(new[] { 5, 12, 12, 3 }, spiking.LayerSizes);
        }

        [Fact]
        public void ConvertedLayersUseTheMatchingSpikeFunctions()
        {
            IMatrixNetwork network = NetworkFactory.BuildMatrixFFN(Settings(output: ActivationTypes.HyperbolicTangens),
                                                                   Hidden(ActivationTypes.RELU), Hidden(ActivationTypes.LeakyRELU));
            ISpikingNetwork spiking = SpikingNetworkConverter.Convert(network, Options());

            Assert.IsType<IntegrateAndFireActivation>(spiking.GetActivation(0));
            Assert.IsType<IntegrateAndFireActivation>(spiking.GetActivation(1));
            Assert.IsType<IntegrateAndFireActivation>(spiking.GetActivation(2));
            Assert.Equal(0.01f, Assert.IsType<SignedIntegrateAndFireActivation>(spiking.GetActivation(3)).NegativeSlope);
            Assert.Equal(0.01f, Assert.IsType<SignedIntegrateAndFireActivation>(spiking.GetActivation(4)).NegativeSlope);
            Assert.Equal(2, Assert.IsType<SigmoidRateActivation>(spiking.GetActivation(5)).Gain);
            Assert.Equal(64, SpikingNetworkConverter.Convert(network, new SpikingConversionOptions { CalibrationInputs = Samples(10, 1) }).TimeSteps);
        }

        [Fact]
        public void NeuronNetworksAndMatrixNetworksConvertAlike()
        {
            INetwork neurons = NetworkFactory.BuildFFN(Settings(), Hidden(ActivationTypes.RELU));
            IMatrixNetwork matrix = NetworkFactory.BuildMatrixFFN(Settings(), Hidden(ActivationTypes.RELU));

            ISpikingNetwork fromNeurons = SpikingNetworkConverter.Convert(neurons, Options());
            ISpikingNetwork fromMatrix = SpikingNetworkConverter.Convert(matrix, Options());

            foreach (float[] input in Samples(10, 4))
            {
                Assert.Equal(fromMatrix.Run(input, 64).OutputSpikes, fromNeurons.Run(input, 64).OutputSpikes);
                float[] a = fromMatrix.Run(input, 64).Outputs.ToArray(), b = fromNeurons.Run(input, 64).Outputs.ToArray();
                for (int i = 0; i < a.Length; i++) Assert.Equal(a[i], b[i], 4);
            }

            Assert.True(Error(neurons, fromNeurons, Samples(20, 5), 512) < 0.02f);
        }

        [Fact]
        public void InputNormalizationInputActivationAndSoftMaxAreKept()
        {
            IMatrixNetwork network = NetworkFactory.BuildMatrixFFN(Settings(normalize: true, softMax: true, input: ActivationTypes.SoftSign),
                                                                   Hidden(ActivationTypes.RELU));
            ISpikingNetwork spiking = SpikingNetworkConverter.Convert(network, Options());

            Assert.True(Error(network, spiking, Samples(30, 6), 512) < 0.02f);
            Assert.Equal(1f, spiking.Run(Samples(1, 7)[0], 64).Outputs.Sum(), 4);
        }

        [Fact]
        public void SignedInputsFireNegativeSpikes()
        {
            IMatrixNetwork network = NetworkFactory.BuildMatrixFFN(Settings(), Hidden(ActivationTypes.RELU));
            SpikingConversionOptions options = Options();
            options.SignedInputs = true;
            ISpikingNetwork spiking = SpikingNetworkConverter.Convert(network, options);

            Assert.Equal(1, Assert.IsType<SignedIntegrateAndFireActivation>(spiking.GetActivation(0)).NegativeSlope);
            Assert.True(Error(network, spiking, Samples(30, 8), 512) < 0.02f);
        }

        [Fact]
        public void GraphsConvertWithoutBuildingTheNetwork()
        {
            // 0, 1 inputs, 2 bias, 3 and 4 ReLU hidden (4 fed by 3), 5 sigmoid output fed by everything
            var settings = new GraphNetworkSettings
            {
                Nodes =
                {
                    new GraphNode(0, NodeType.INPUT), new GraphNode(1, NodeType.INPUT), new GraphNode(2, NodeType.BIAS),
                    new GraphNode(3, NodeType.HIDDEN), new GraphNode(4, NodeType.HIDDEN, ActivationTypes.LeakyRELU), new GraphNode(5, NodeType.OUTPUT)
                },
                Connections =
                {
                    new GraphConnection(0, 0, 3, 0.8f, true), new GraphConnection(1, 1, 3, -0.6f, true), new GraphConnection(2, 3, 4, 1.5f, true),
                    new GraphConnection(3, 0, 5, 0.5f, true), new GraphConnection(4, 3, 5, -1, true), new GraphConnection(5, 4, 5, 0.7f, true),
                    new GraphConnection(6, 2, 5, 0.25f, true), new GraphConnection(7, 2, 3, 0.1f, true)
                },
                HiddenActivationType = ActivationTypes.RELU,
                OutputActivationType = ActivationTypes.Sigmoid
            };

            IGraphNetwork graph = NetworkFactory.BuildGraph(settings);
            ISpikingNetwork spiking = SpikingNetworkConverter.ConvertGraph(settings, new SpikingConversionOptions
            {
                CalibrationInputs = Samples(200, 1, 2),
                Percentile = 100
            });

            Assert.True(Error(graph, spiking, Samples(30, 9, 2), 512) < 0.02f);
        }

        [Fact]
        public void UnsupportedNetworksAreRejected()
        {
            IMatrixNetwork gelu = NetworkFactory.BuildMatrixFFN(Settings(), Hidden(ActivationTypes.GELU));
            Assert.Throws<NotSupportedException>(() => SpikingNetworkConverter.Convert(gelu, Options()));

            IMatrixNetwork network = NetworkFactory.BuildMatrixFFN(Settings(), Hidden(ActivationTypes.RELU));
            Assert.Throws<ArgumentException>(() => SpikingNetworkConverter.Convert(network, new SpikingConversionOptions()));
            Assert.Throws<ArgumentException>(() => SpikingNetworkConverter.Convert(network, new SpikingConversionOptions { CalibrationInputs = Samples(5, 1, 4) }));
            Assert.Throws<ArgumentException>(() => SpikingNetworkConverter.Convert(network, new SpikingConversionOptions { CalibrationInputs = Samples(5, 1), Percentile = 0 }));
            Assert.Throws<ArgumentNullException>(() => SpikingNetworkConverter.Convert(network, null));
        }

        [Fact]
        public void SpikesAreCheaperWhereActivationsAreZero()
        {
            // ReLU networks are sparse: neurons with a negative input never fire and cost nothing
            IMatrixNetwork network = NetworkFactory.BuildMatrixFFN(Settings(), Hidden(ActivationTypes.RELU));
            ISpikingNetwork spiking = SpikingNetworkConverter.Convert(network, Options());

            SpikingRun run = spiking.Run(Samples(1, 10)[0], 64);
            Assert.True(run.Spikes < 64L * spiking.NeuronCount);
            Assert.True(run.SynapticEvents > 0);
        }
    }
}
