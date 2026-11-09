using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Spiking;
using ANN.Net.Spiking;
using Xunit;

namespace ANN.Net.Tests.Spiking
{
    /// <summary>
    /// How spikes travel through a spiking network, and how STDP changes its synapses
    /// </summary>
    public class SpikingNetworkTests
    {
        /// <summary>
        /// Input encoding for exact timing: a neuron fires once, in the step given as its input
        /// </summary>
        private sealed class ScheduledActivation : ISpikingActivation
        {
            public int Step(ref SpikingNeuronState neuron, float current)
            {
                neuron.Steps++;
                return neuron.Steps == (int)current ? 1 : 0;
            }
        }

        private static SpikingNetwork Network(int[] sizes, ISpikingActivation[] activations, IEnumerable<SpikingSynapse> synapses,
                                              bool[] learns = null, float[] bias = null, StdpOptions plasticity = null, InputScaling scaling = null)
        {
            int outputs = sizes[^1];
            return new SpikingNetwork(sizes, activations, learns ?? sizes.Select(_ => false).ToArray(), bias, synapses, scaling, null,
                                      SpikingReadout.Rates(sizes.Sum() - outputs, outputs), 10, plasticity);
        }

        private static ISpikingActivation[] Fire(int layers) => Enumerable.Range(0, layers).Select(_ => (ISpikingActivation)new IntegrateAndFireActivation()).ToArray();

        [Fact]
        public void SpikesTravelThroughAllLayersInOneStep()
        {
            var network = Network(new[] { 1, 1, 1 }, Fire(3), new[] { new SpikingSynapse(0, 1, 1), new SpikingSynapse(1, 2, 1) });

            SpikingRun run = network.Run(new[] { 1f }, 1);

            Assert.Equal(new[] { 1 }, run.OutputSpikes);
            Assert.Equal(3, run.Spikes);
            Assert.Equal(2, run.SynapticEvents);
        }

        [Fact]
        public void SynapsesWithinALayerDeliverInTheNextStep()
        {
            var network = Network(new[] { 1, 2 }, Fire(2), new[] { new SpikingSynapse(0, 1, 1), new SpikingSynapse(1, 2, 1) });

            Assert.Equal(new[] { 1, 0 }, network.Run(new[] { 1f }, 1).OutputSpikes);
            Assert.Equal(new[] { 2, 1 }, network.Run(new[] { 1f }, 2).OutputSpikes);
        }

        [Fact]
        public void NegativeSpikesSubtractTheWeight()
        {
            ISpikingActivation signed = new SignedIntegrateAndFireActivation(negativeSlope: 1);
            var network = Network(new[] { 1, 1 }, new[] { signed, signed }, new[] { new SpikingSynapse(0, 1, 0.5f) });

            // the input fires a negative spike every step, the output every second step
            SpikingRun run = network.Run(new[] { -1f }, 10);

            Assert.Equal(new[] { -5 }, run.OutputSpikes);
            Assert.Equal(-0.5f, run.Outputs[0]);
        }

        [Fact]
        public void SilentNeuronsCostNoSynapticWork()
        {
            var synapses = Enumerable.Range(0, 4).SelectMany(i => Enumerable.Range(4, 3).Select(t => new SpikingSynapse(i, t, 0.3f)));
            var network = Network(new[] { 4, 3 }, Fire(2), synapses);

            SpikingRun silent = network.Run(new float[4], 20);
            Assert.Equal(0, silent.Spikes);
            Assert.Equal(0, silent.SynapticEvents);

            // one input firing every second step: 10 spikes of 3 synapses
            SpikingRun one = network.Run(new[] { 0.5f, 0, 0, 0 }, 20);
            Assert.Equal(30, one.SynapticEvents);
            Assert.Equal(10, one.Spikes - one.OutputSpikes.Sum());
        }

        [Fact]
        public void OutputsAreFiringRates()
        {
            var network = Network(new[] { 1, 1 }, Fire(2), new[] { new SpikingSynapse(0, 1, 0.5f) });

            // the input fires every step, the output every second
            Assert.Equal(new[] { 0.5f }, network.Run(new[] { 1f }, 40).Outputs);
            network.TimeSteps = 40;
            Assert.Equal(new[] { 0.5f }, network.Propagate(new[] { 1f }));
        }

        [Fact]
        public void BiasIsAddedEveryStep()
        {
            var network = Network(new[] { 1, 1 }, Fire(2), Array.Empty<SpikingSynapse>(), bias: new[] { 0f, 0.25f });

            Assert.Equal(new[] { 10 }, network.Run(new[] { 0f }, 40).OutputSpikes);
        }

        [Fact]
        public void ConsecutiveAndScatteredTargetsGiveTheSameResult()
        {
            // 3 inputs to 37 neurons: in order the weights of a source are added with SIMD, shuffled one by one
            var random = new Random(5);
            var synapses = Enumerable.Range(0, 3).SelectMany(i => Enumerable.Range(3, 37)
                                     .Select(t => new SpikingSynapse(i, t, (float)((random.NextDouble() * 2) - 0.5)))).ToList();
            var shuffled = synapses.OrderBy(_ => random.Next()).ToList();

            float[] input = { 0.7f, 0.3f, 1f };
            SpikingRun ordered = Network(new[] { 3, 37 }, Fire(2), synapses).Run(input, 50);
            SpikingRun scattered = Network(new[] { 3, 37 }, Fire(2), shuffled).Run(input, 50);

            Assert.Equal(ordered.OutputSpikes, scattered.OutputSpikes);
            Assert.True(ordered.OutputSpikes.Sum() > 100);
        }

        [Fact]
        public void ActivationsCanBeReplaced()
        {
            var network = Network(new[] { 1, 1 }, Fire(2), new[] { new SpikingSynapse(0, 1, 1) });
            Assert.Equal(new[] { 20 }, network.Run(new[] { 1f }, 20).OutputSpikes);

            network.SetActivation(1, new IntegrateAndFireActivation(threshold: 4));
            Assert.Equal(new[] { 5 }, network.Run(new[] { 1f }, 20).OutputSpikes);
            Assert.IsType<IntegrateAndFireActivation>(network.GetActivation(1));

            network.SetActivation(0, new PoissonActivation(seed: 1));
            Assert.InRange(network.Run(new[] { 0.5f }, 400).OutputSpikes[0], 30, 70);

            Assert.Throws<ArgumentNullException>(() => network.SetActivation(1, null));
            Assert.Throws<ArgumentOutOfRangeException>(() => network.SetActivation(2, new SigmoidRateActivation()));
        }

        [Fact]
        public void InputsAreScaled()
        {
            var network = Network(new[] { 1, 1 }, Fire(2), new[] { new SpikingSynapse(0, 1, 1) }, scaling: InputScaling.Range(0, 255, 1));

            Assert.Equal(new[] { 0.5f }, network.Run(new[] { 127.5f }, 20).Outputs);
        }

        [Fact]
        public void InvalidSynapsesAreRejected()
        {
            Assert.Throws<ArgumentException>(() => Network(new[] { 1, 1 }, Fire(2), new[] { new SpikingSynapse(1, 0, 1) }));
            Assert.Throws<ArgumentException>(() => Network(new[] { 1, 1 }, Fire(2), new[] { new SpikingSynapse(0, 2, 1) }));
            Assert.Throws<ArgumentException>(() => Network(new[] { 1, 1 }, Fire(2), new[] { new SpikingSynapse(0, 1, float.NaN) }));

            var network = Network(new[] { 2, 1 }, Fire(2), new[] { new SpikingSynapse(0, 2, 1) });
            Assert.Throws<ArgumentException>(() => network.Run(new[] { 1f }, 5));
            Assert.Throws<ArgumentOutOfRangeException>(() => network.Run(new[] { 1f, 1f }, 0));
        }

        [Fact]
        public void BuildCreatesTheLayersAndSynapses()
        {
            ISpikingNetwork network = NetworkFactory.BuildSpiking(new SpikingNetworkSettings
            {
                InputCount = 6,
                Seed = 4,
                Layers =
                {
                    new SpikingLayerSettings { NeuronsCount = 5, MinimumWeight = 0.2f, MaximumWeight = 0.4f, LateralInhibition = 1.5f },
                    new SpikingLayerSettings { NeuronsCount = 3, ConnectionProbability = 0.5 }
                }
            });

            Assert.Equal(new[] { 6, 5, 3 }, network.LayerSizes);
            Assert.Equal(14, network.NeuronCount);
            Assert.Equal(6, network.InputCount);
            Assert.Equal(3, network.OutputCount);

            IReadOnlyList<SpikingSynapse> synapses = network.GetSynapses();
            Assert.Equal(network.SynapseCount, synapses.Count);
            Assert.Equal(30, synapses.Count(s => s.Source < 6));
            Assert.All(synapses.Where(s => s.Source < 6), s => Assert.InRange(s.Weight, 0.2f, 0.4f));
            Assert.Equal(20, synapses.Count(s => s.Source >= 6 && s.Target < 11));
            Assert.All(synapses.Where(s => s.Source >= 6 && s.Target < 11), s => Assert.Equal(-1.5f, s.Weight));
            Assert.InRange(synapses.Count(s => s.Target >= 11), 5, 12);

            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildSpiking(new SpikingNetworkSettings { InputCount = 2 }));
            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildSpiking(new SpikingNetworkSettings
            {
                InputCount = 2,
                Layers = { new SpikingLayerSettings { NeuronsCount = 2, ConnectionProbability = 2 } }
            }));
        }

        [Fact]
        public void GraphsOfAnyTopologyCanBeBuilt()
        {
            // inputs 0, 1; layer 1: neuron 2 (fed by 0) and 3 (fed by 0 and 1, bias 0.25); layer 2: neuron 4 fed by 2 and
            // by input 1 directly; outputs 4 and 3
            ISpikingNetwork network = NetworkFactory.BuildSpiking(new SpikingGraphSettings
            {
                Layers = { new SpikingGraphLayer(2), new SpikingGraphLayer(2), new SpikingGraphLayer(1, new IntegrateAndFireActivation(threshold: 2)) },
                Synapses =
                {
                    new SpikingSynapse(0, 2, 1), new SpikingSynapse(0, 3, 0.25f), new SpikingSynapse(1, 3, 0.25f),
                    new SpikingSynapse(2, 4, 1), new SpikingSynapse(1, 4, 1)
                },
                Bias = new[] { 0f, 0f, 0f, 0.25f, 0f },
                Outputs = { 4, 3 },
                TimeSteps = 40
            });

            Assert.Equal(new[] { 2, 2, 1 }, network.LayerSizes);
            Assert.Equal(5, network.SynapseCount);

            // 2 fires every step; 3 gets 0.25 + 0.25 + bias 0.25 per step and fires 30 times; 4 gets 2 per step against threshold 2
            SpikingRun run = network.Run(new[] { 1f, 1f }, 40);
            Assert.Equal(40, run.OutputSpikes[0]);
            Assert.Equal(30, run.OutputSpikes[1]);
            Assert.Equal(new[] { 1f, 0.75f }, run.Outputs);
        }

        [Fact]
        public void CyclesGiveTheNetworkMemory()
        {
            // input 0.25 fires every fourth step; a neuron that excites itself fires every step from its first spike on
            SpikingGraphSettings Settings(float loop) => new SpikingGraphSettings
            {
                Layers = { new SpikingGraphLayer(1), new SpikingGraphLayer(1) },
                Synapses = { new SpikingSynapse(0, 1, 1), new SpikingSynapse(1, 1, loop) }
            };

            Assert.Equal(new[] { 10 }, NetworkFactory.BuildSpiking(Settings(0)).Run(new[] { 0.25f }, 40).OutputSpikes);
            Assert.Equal(new[] { 37 }, NetworkFactory.BuildSpiking(Settings(1)).Run(new[] { 0.25f }, 40).OutputSpikes);
        }

        [Fact]
        public void InvalidGraphsAreRejected()
        {
            SpikingGraphSettings Valid() => new SpikingGraphSettings
            {
                Layers = { new SpikingGraphLayer(1), new SpikingGraphLayer(1) },
                Synapses = { new SpikingSynapse(0, 1, 1) }
            };

            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildSpiking(new SpikingGraphSettings { Layers = { new SpikingGraphLayer(1) } }));
            Assert.Throws<ArgumentOutOfRangeException>(() => new SpikingGraphLayer(0));

            var outputs = Valid();
            outputs.Outputs.Add(5);
            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildSpiking(outputs));

            var bias = Valid();
            bias.Bias = new[] { 0f };
            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildSpiking(bias));
            bias.Bias = new[] { 1f, 0f };
            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildSpiking(bias));

            var synapse = Valid();
            synapse.Synapses.Add(new SpikingSynapse(1, 0, 1));
            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildSpiking(synapse));
        }

        private static float StdpWeight(float sourceStep, float driverStep)
        {
            var plasticity = new StdpOptions
            {
                PotentiationRate = 0.1f,
                DepressionRate = 0.1f,
                PotentiationTimeConstant = 10,
                DepressionTimeConstant = 10
            };

            // the driver makes the target fire; the weight of the weak source is the one that learns
            var network = Network(new[] { 2, 1 }, new ISpikingActivation[] { new ScheduledActivation(), new IntegrateAndFireActivation() },
                                  new[] { new SpikingSynapse(0, 2, 0.2f), new SpikingSynapse(1, 2, 1f) },
                                  learns: new[] { false, true }, plasticity: plasticity);

            network.Learn(new[] { sourceStep, driverStep });
            return network.GetSynapses().Single(s => s.Source == 0).Weight;
        }

        [Fact]
        public void SourceFiringBeforeTheTargetStrengthensTheSynapse()
        {
            // fired 2 steps before the target: + rate * exp(-2 / tau) * (maximum - weight)
            Assert.Equal(0.2f + (0.1f * MathF.Exp(-0.2f) * 0.8f), StdpWeight(sourceStep: 3, driverStep: 5), 5);
        }

        [Fact]
        public void SourceFiringAfterTheTargetWeakensTheSynapse()
        {
            // fired 2 steps after the target: - rate * exp(-2 / tau) * (weight - minimum)
            Assert.Equal(0.2f - (0.1f * MathF.Exp(-0.2f) * 0.2f), StdpWeight(sourceStep: 7, driverStep: 5), 5);
        }

        [Fact]
        public void DistantSpikesDoNotChangeTheSynapse()
        {
            Assert.Equal(0.2f, StdpWeight(sourceStep: 2, driverStep: 1000), 5);
        }

        [Fact]
        public void RunWithoutLearningKeepsTheWeights()
        {
            var network = Network(new[] { 2, 1 }, new ISpikingActivation[] { new ScheduledActivation(), new IntegrateAndFireActivation() },
                                  new[] { new SpikingSynapse(0, 2, 0.2f), new SpikingSynapse(1, 2, 1f) }, learns: new[] { false, true });

            network.Run(new[] { 3f, 5f }, 10);

            Assert.Equal(0.2f, network.GetSynapses().Single(s => s.Source == 0).Weight);
        }

        [Fact]
        public void CompetingNeuronsLearnDifferentPatterns()
        {
            float[] first = { 0.5f, 0.5f, 0.5f, 0.5f, 0, 0, 0, 0 };
            float[] second = { 0, 0, 0, 0, 0.5f, 0.5f, 0.5f, 0.5f };

            // Poisson inputs, so that the two neurons never receive exactly the same spikes; adaptation and lateral
            // inhibition make them compete
            ISpikingNetwork network = NetworkFactory.BuildSpiking(new SpikingNetworkSettings
            {
                InputCount = 8,
                Seed = 0,
                TimeSteps = 40,
                InputActivation = new PoissonActivation(seed: 0),
                Plasticity = new StdpOptions
                {
                    PotentiationRate = 0.05f,
                    DepressionRate = 0.03f,
                    PotentiationTimeConstant = 4,
                    DepressionTimeConstant = 4,
                    IncomingWeightSum = 3
                },
                Layers =
                {
                    new SpikingLayerSettings
                    {
                        NeuronsCount = 2,
                        MinimumWeight = 0.2f,
                        MaximumWeight = 0.5f,
                        LateralInhibition = 3,
                        Activation = new IntegrateAndFireActivation(leak: 0.9f, thresholdAdaptation: 0.05f, adaptationDecay: 0.999f)
                    }
                }
            });

            // the responses are measured without adaptation, so that one input does not change the response to the next
            (int[] First, int[] Second) Responses()
            {
                ISpikingActivation learning = network.GetActivation(1);
                network.SetActivation(1, new IntegrateAndFireActivation(leak: 0.9f));
                var result = (network.Run(first, 40).OutputSpikes.ToArray(), network.Run(second, 40).OutputSpikes.ToArray());
                network.SetActivation(1, learning);
                return result;
            }

            // with its initial random weights neuron 0 wins both patterns
            var (before1, before2) = Responses();
            Assert.True(before1[0] > before1[1] && before2[0] > before2[1], $"{before1[0]}/{before1[1]} {before2[0]}/{before2[1]}");

            for (int round = 0; round < 30; round++)
            {
                network.Learn(first);
                network.Learn(second);
            }

            var (after1, after2) = Responses();
            int winner1 = after1[0] > after1[1] ? 0 : 1, winner2 = after2[0] > after2[1] ? 0 : 1;
            Assert.NotEqual(winner1, winner2);
            Assert.True(after1[winner1] > 5 * after1[1 - winner1], $"first pattern: {after1[0]} / {after1[1]}");
            Assert.True(after2[winner2] > 5 * after2[1 - winner2], $"second pattern: {after2[0]} / {after2[1]}");

            // each winner's synapses from its pattern's inputs grew, the others faded
            IReadOnlyList<SpikingSynapse> synapses = network.GetSynapses();
            float From(int neuron, int firstInput) => synapses.Where(s => s.Target == 8 + neuron && s.Source >= firstInput && s.Source < firstInput + 4).Sum(s => s.Weight);
            Assert.True(From(winner1, 0) > 5 * From(winner1, 4));
            Assert.True(From(winner2, 4) > 5 * From(winner2, 0));
            Assert.Equal(3, From(winner1, 0) + From(winner1, 4), 3);
        }
    }
}
