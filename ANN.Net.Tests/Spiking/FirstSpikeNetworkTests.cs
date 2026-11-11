using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Abstractions.Spiking;
using ANN.Net.IdentityGenerators;
using Xunit;

namespace ANN.Net.Tests.Spiking
{
    /// <summary>
    /// Time-to-first-spike networks: every neuron fires at most once, earlier for a larger value, and the converted
    /// network computes what the trained one computes up to rounding to 1 / window of every layer's scale
    /// </summary>
    public class FirstSpikeNetworkTests
    {
        private static NetworkSettings Settings(bool softMax = false, ActivationTypes output = ActivationTypes.Sigmoid) => new NetworkSettings
        {
            Seed = 3,
            IdentityGenerator = new IncrementalGenerator(),
            OutputApplySoftMax = softMax,
            InputSettings = new InputLayerSettings { NeuronsCount = 5, HasBiasNeuron = true, ActivationType = ActivationTypes.Identity },
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

        private static float Error(INetwork network, IFirstSpikeNetwork spiking, IEnumerable<float[]> samples, int window)
        {
            float error = 0;
            foreach (float[] input in samples)
            {
                Quad[] expected = network.Propagate(input.Select(x => (Quad)x).ToArray());
                IReadOnlyList<float> actual = spiking.Run(input, window).Outputs;
                for (int i = 0; i < expected.Length; i++) error = Math.Max(error, Math.Abs(expected[i] - actual[i]));
            }

            return error;
        }

        private static SpikingConversionOptions Options(int window = 256) => new SpikingConversionOptions
        {
            CalibrationInputs = Samples(300, 1),
            Percentile = 100,
            TimeSteps = window
        };

        [Theory]
        [InlineData(ActivationTypes.RELU)]
        [InlineData(ActivationTypes.LeakyRELU)]
        [InlineData(ActivationTypes.Identity)]
        [InlineData(ActivationTypes.Sigmoid)]
        [InlineData(ActivationTypes.HyperbolicTangens)]
        [InlineData(ActivationTypes.GELU)]
        public void ConvertedNetworkComputesWhatTheNetworkComputes(ActivationTypes activation)
        {
            IMatrixNetwork network = NetworkFactory.BuildMatrixFFN(Settings(), Hidden(activation));
            IFirstSpikeNetwork spiking = SpikingNetworkConverter.ConvertToFirstSpike(network, Options());
            List<float[]> samples = Samples(40, 2);

            float coarse = Error(network, spiking, samples, 4), fine = Error(network, spiking, samples, 1024);

            Assert.True(fine < 0.01f, $"error with a window of 1024 steps {fine}");
            Assert.True(fine < coarse, $"error with 4 steps {coarse}, with 1024 steps {fine}");
            Assert.Equal(new[] { 5, 12, 12, 3 }, spiking.LayerSizes);
        }

        [Fact]
        public void EveryNeuronFiresAtMostOnceInItsLayersWindow()
        {
            IMatrixNetwork network = NetworkFactory.BuildMatrixFFN(Settings(), Hidden(ActivationTypes.RELU));
            IFirstSpikeNetwork spiking = SpikingNetworkConverter.ConvertToFirstSpike(network, Options(16));

            foreach (float[] input in Samples(20, 3))
            {
                SpikingRun run = spiking.Run(input, 16);
                IReadOnlyList<FirstSpike> spikes = spiking.GetLastSpikes();

                Assert.Equal(spikes.Count, spikes.Select(s => s.Neuron).Distinct().Count());
                Assert.Equal(run.Spikes, spikes.Count);
                Assert.True(run.SynapticEvents <= spiking.SynapseCount);
                Assert.Equal(3 * 16, run.TimeSteps);

                // layer d fires in steps [16 d, 16 d + 16)
                int[] layerStart = { 0, 5, 17, 29, 32 };
                foreach (FirstSpike spike in spikes)
                {
                    int layer = Array.FindLastIndex(layerStart, s => s <= spike.Neuron);
                    Assert.InRange(spike.Step, 16 * layer, (16 * layer) + 15);
                }

                Assert.Equal(spikes.OrderBy(s => s.Step).Select(s => s.Step), spikes.Select(s => s.Step));
            }
        }

        [Fact]
        public void LargerValuesFireEarlier()
        {
            // inputs 0, 1 from [0, 1]; hidden ReLU h = x0 + x1; output identity o = h; calibration maximum of h is 2
            var settings = new GraphNetworkSettings
            {
                Nodes = { new GraphNode(0, NodeType.INPUT), new GraphNode(1, NodeType.INPUT), new GraphNode(2, NodeType.HIDDEN), new GraphNode(3, NodeType.OUTPUT) },
                Connections = { new GraphConnection(0, 0, 2, 1, true), new GraphConnection(1, 1, 2, 1, true), new GraphConnection(2, 2, 3, 1, true) },
                HiddenActivationType = ActivationTypes.RELU,
                OutputActivationType = ActivationTypes.Identity
            };

            IFirstSpikeNetwork spiking = SpikingNetworkConverter.ConvertGraphToFirstSpike(settings, new SpikingConversionOptions
            {
                CalibrationInputs = new[] { new[] { 0f, 0f }, new[] { 1f, 1f } },
                Percentile = 100,
                TimeSteps = 8
            });

            SpikingRun run = spiking.Run(new[] { 0.5f, 0.25f }, 8);

            // 0.5 fires in step 8 - 4, 0.25 in step 8 - 2; h = 0.75 is 3/8 of 2: step 8 + 8 - 3; o = 0.75 likewise in the third window
            Assert.Equal(new[] { new FirstSpike(0, 4, 1), new FirstSpike(1, 6, 1), new FirstSpike(2, 13, 1), new FirstSpike(3, 21, 1) }, spiking.GetLastSpikes());
            Assert.Equal(new[] { 0.75f }, run.Outputs);
            Assert.Equal(new[] { 1 }, run.OutputSpikes);
            Assert.Equal(2 + 1, run.SynapticEvents);
        }

        [Fact]
        public void ZeroValuesDoNotFire()
        {
            var settings = new GraphNetworkSettings
            {
                Nodes = { new GraphNode(0, NodeType.INPUT), new GraphNode(1, NodeType.INPUT), new GraphNode(2, NodeType.HIDDEN), new GraphNode(3, NodeType.OUTPUT) },
                Connections = { new GraphConnection(0, 0, 2, 1, true), new GraphConnection(1, 1, 2, -1, true), new GraphConnection(2, 2, 3, 1, true) },
                HiddenActivationType = ActivationTypes.RELU,
                OutputActivationType = ActivationTypes.Identity
            };
            IFirstSpikeNetwork spiking = SpikingNetworkConverter.ConvertGraphToFirstSpike(settings, new SpikingConversionOptions
            {
                CalibrationInputs = new[] { new[] { 0f, 0f }, new[] { 1f, 1f } }
            });

            SpikingRun silent = spiking.Run(new[] { 0f, 0f }, 64);
            Assert.Equal(0, silent.Spikes);
            Assert.Equal(0, silent.SynapticEvents);

            // h = ReLU(0.2 - 0.6) = 0: only the inputs fire
            SpikingRun negative = spiking.Run(new[] { 0.2f, 0.6f }, 64);
            Assert.Equal(2, negative.Spikes);
            Assert.Equal(2, negative.SynapticEvents);
            Assert.Equal(new[] { 0f }, negative.Outputs);
        }

        [Fact]
        public void NegativeValuesFireNegativeSpikes()
        {
            IMatrixNetwork network = NetworkFactory.BuildMatrixFFN(Settings(), Hidden(ActivationTypes.Identity));
            SpikingConversionOptions options = Options();
            options.SignedInputs = true;
            IFirstSpikeNetwork spiking = SpikingNetworkConverter.ConvertToFirstSpike(network, options);

            spiking.Run(new[] { -1.5f, 0.5f, -0.2f, 1.8f, -2f }, 64);
            IReadOnlyList<FirstSpike> spikes = spiking.GetLastSpikes();

            Assert.Contains(spikes, s => s.Neuron < 5 && s.Sign < 0);
            Assert.Contains(spikes, s => s.Neuron >= 5 && s.Sign < 0);
            Assert.True(Error(network, spiking, Samples(30, 8), 1024) < 0.01f);
        }

        [Fact]
        public void TheWorkDoesNotGrowWithTheWindow()
        {
            IMatrixNetwork network = NetworkFactory.BuildMatrixFFN(Settings(), Hidden(ActivationTypes.RELU));
            IFirstSpikeNetwork spiking = SpikingNetworkConverter.ConvertToFirstSpike(network, Options());
            float[] input = Samples(1, 9)[0];

            long small = spiking.Run(input, 64).SynapticEvents, large = spiking.Run(input, 65536).SynapticEvents;
            Assert.InRange(large, small, spiking.SynapseCount);

            spiking.Window = 32;
            Assert.Equal(spiking.Run(input, 32).Outputs, spiking.Propagate(input));
            Assert.Throws<ArgumentOutOfRangeException>(() => spiking.Window = 0);
        }

        [Fact]
        public void NeuronNetworksAndSoftMaxConvert()
        {
            INetwork neurons = NetworkFactory.BuildFFN(Settings(), Hidden(ActivationTypes.RELU));
            IFirstSpikeNetwork spiking = SpikingNetworkConverter.ConvertToFirstSpike(neurons, Options());

            Assert.True(Error(neurons, spiking, Samples(20, 5), 1024) < 0.01f);
            Assert.Equal(spiking.SynapseCount, spiking.GetSynapses().Count);

            // the network divides its outputs by their sum
            INetwork softMax = NetworkFactory.BuildFFN(Settings(softMax: true), Hidden(ActivationTypes.RELU));
            Assert.Equal(1f, SpikingNetworkConverter.ConvertToFirstSpike(softMax, Options()).Propagate(Samples(1, 6)[0]).Sum(), 4);
        }

        [Fact]
        public void GraphsWithConnectionsThatSkipLayersConvert()
        {
            // 0, 1 inputs, 2 bias, 3 ReLU and 4 leaky ReLU hidden (4 fed by 3), 5 sigmoid output fed by everything
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
            IFirstSpikeNetwork spiking = SpikingNetworkConverter.ConvertGraphToFirstSpike(settings, new SpikingConversionOptions
            {
                CalibrationInputs = Samples(200, 1, 2),
                Percentile = 100
            });

            Assert.True(Error(graph, spiking, Samples(30, 9, 2), 1024) < 0.01f);
        }

        [Fact]
        public void UnsupportedNetworksAreRejected()
        {
            IMatrixNetwork network = NetworkFactory.BuildMatrixFFN(Settings(), Hidden(ActivationTypes.RELU));
            Assert.Throws<ArgumentException>(() => SpikingNetworkConverter.ConvertToFirstSpike(network, new SpikingConversionOptions()));
            Assert.Throws<ArgumentNullException>(() => SpikingNetworkConverter.ConvertToFirstSpike(null, Options()));

            IFirstSpikeNetwork spiking = SpikingNetworkConverter.ConvertToFirstSpike(network, Options());
            Assert.Throws<ArgumentException>(() => spiking.Run(new float[4], 8));
            Assert.Throws<ArgumentOutOfRangeException>(() => spiking.Run(new float[5], 0));
        }
    }
}
