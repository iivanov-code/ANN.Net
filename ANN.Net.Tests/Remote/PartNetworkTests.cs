using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Connections;
using ANN.Net.Neurons;
using ANN.Net.Remote;
using Xunit;

namespace ANN.Net.Tests.Remote
{
    /// <summary>
    /// The parts of a breadth-split section wired to each other directly, compared with one plain network of
    /// the same section whose synapses carry the same weights
    /// </summary>
    public class PartNetworkTests
    {
        private const int INPUTS = 3;

        private static readonly RecurrentLayerSettings[] LAYERS =
        {
            SplitHelpers.Layer(4, ActivationTypes.HyperbolicTangens, bias: true),
            SplitHelpers.Layer(5, ActivationTypes.Sigmoid),
            SplitHelpers.Layer(3, ActivationTypes.Sigmoid, bias: true)
        };

        /// <summary>
        /// Deterministic weight for the synapse from neuron <paramref name="from"/> to neuron <paramref name="to"/> of section layer <paramref name="layer"/>
        /// </summary>
        private static float Weight(int layer, int from, int to)
        {
            return (((layer * 31 + from * 7 + to * 13) % 17) - 8) / 10f;
        }

        private static NetworkPartition Partition(int parts)
        {
            NetworkSettings whole = SplitHelpers.Whole(2, 2, ActivationTypes.Identity, inputBias: false);
            RecurrentLayerSettings[] hidden = new[] { SplitHelpers.Layer(INPUTS, ActivationTypes.Sigmoid, bias: true) }.Concat(LAYERS).ToArray();
            return RemoteNetworkFactory.Partition(whole, hidden, 1, 3, parts);
        }

        /// <summary>
        /// The same section as one network: identity inputs with the bias, the hidden layers, the last layer as activated outputs
        /// </summary>
        private static Network PlainSection()
        {
            NetworkSettings settings = new NetworkSettings
            {
                IdentityGenerator = new IdentityGenerators.IncrementalGenerator(),
                LossType = LossFunctionTypes.Squared,
                OutputApplySoftMax = false,
                InputSettings = new InputLayerSettings { NeuronsCount = INPUTS, ActivationType = ActivationTypes.Identity, HasBiasNeuron = true },
                OutputSettings = new OutputLayerSettings { NeuronsCount = LAYERS[2].NeuronsCount, ActivationType = LAYERS[2].ActivationType }
            };

            return (Network)NetworkFactory.BuildFFN(settings, null, new[] { LAYERS[0], LAYERS[1] });
        }

        /// <summary>
        /// Assigns <see cref="Weight"/> to every synapse of the plain network and returns them by (layer, from, to)
        /// </summary>
        private static Dictionary<(int, int, int), BaseSynapse> AssignPlainWeights(Network network)
        {
            List<INeuron[]> layers = new List<INeuron[]> { network.InputNeurons.Cast<INeuron>().ToArray() };
            layers.AddRange(network.HiddenLayers.Select(layer => layer.Cast<INeuron>().ToArray()));

            Dictionary<(int, int, int), BaseSynapse> synapses = new Dictionary<(int, int, int), BaseSynapse>();
            for (int layer = 0; layer < layers.Count; layer++)
            {
                int width = layers[layer].Count(n => !(n is BiasNeuron));
                int index = 0;
                foreach (INeuron neuron in layers[layer])
                {
                    int from = neuron is BiasNeuron ? width : index++;
                    int to = 0;
                    foreach (BaseSynapse synapse in neuron.Outputs.Cast<BaseSynapse>())
                    {
                        synapse.Weight = Weight(layer, from, to);
                        synapses[(layer, from, to)] = synapse;
                        to++;
                    }
                }
            }

            return synapses;
        }

        private sealed class Wiring
        {
            public PartNetwork[] Parts;
            public Quad[][] Outputs;
            public Quad[][] InputGradients;
            public NetworkPartition Partition;

            public Quad[] JoinedOutputs()
            {
                return this.Outputs.SelectMany(slice => slice).ToArray();
            }

            public Quad[] SummedInputGradients()
            {
                Quad[] sum = new Quad[this.Partition.InputWidth];
                foreach (Quad[] partial in this.InputGradients)
                {
                    for (int i = 0; i < sum.Length; i++) sum[i] += partial[i];
                }

                return sum;
            }

            public void Propagate(Quad[] inputs, params int[] order)
            {
                Array.Clear(this.Outputs, 0, this.Outputs.Length);
                foreach (int part in order) this.Parts[part].DeliverInputs(inputs);
                Assert.All(this.Outputs, slice => Assert.NotNull(slice));
            }

            public void Backpropagate(Quad[] outputGradient)
            {
                Array.Clear(this.InputGradients, 0, this.InputGradients.Length);
                int last = this.Partition.LayerCount - 1;
                for (int part = 0; part < this.Parts.Length; part++)
                {
                    Quad[] slice = outputGradient.Skip(this.Partition.SliceStart(last, part)).Take(this.Partition.SliceCount(last, part)).ToArray();
                    this.Parts[part].DeliverOutputGradients(slice);
                }

                Assert.All(this.InputGradients, partial => Assert.NotNull(partial));
            }
        }

        private static Wiring Wire(int parts)
        {
            Wiring wiring = new Wiring
            {
                Partition = Partition(parts),
                Parts = new PartNetwork[parts],
                Outputs = new Quad[parts][],
                InputGradients = new Quad[parts][]
            };

            for (int p = 0; p < parts; p++)
            {
                PartNetwork part = RemoteNetworkFactory.BuildPart(wiring.Partition, p);
                int index = p;
                part.ActivationsReady = (to, layer, values) => wiring.Parts[to].DeliverActivations(index, layer, values);
                part.GradientsReady = (to, layer, values) => wiring.Parts[to].DeliverGradients(index, layer, values);
                part.OutputsReady = values => wiring.Outputs[index] = values;
                part.InputGradientsReady = values => wiring.InputGradients[index] = values;
                wiring.Parts[p] = part;

                foreach (PartSynapse synapse in part.Synapses)
                {
                    synapse.Synapse.Weight = Weight(synapse.Layer, synapse.From, synapse.To);
                }
            }

            return wiring;
        }

        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(3)]
        public void Parts_TogetherHoldEverySynapseOfTheSectionOnce(int parts)
        {
            Wiring wiring = Wire(parts);
            Network plain = PlainSection();
            Dictionary<(int, int, int), BaseSynapse> expected = AssignPlainWeights(plain);

            List<(int, int, int)> actual = wiring.Parts.SelectMany(p => p.Synapses).Select(s => (s.Layer, s.From, s.To)).ToList();

            Assert.Equal(expected.Count, actual.Count);
            Assert.Equal(expected.Keys.OrderBy(k => k), actual.OrderBy(k => k));
        }

        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(3)]
        public void Propagate_MatchesThePlainSection(int parts)
        {
            Wiring wiring = Wire(parts);
            Network plain = PlainSection();
            AssignPlainWeights(plain);
            Quad[] inputs = { 0.5f, -1.25f, 0.75f };

            Quad[] expected = plain.Propagate(inputs);
            wiring.Propagate(inputs, Enumerable.Range(0, parts).ToArray());

            Quad[] actual = wiring.JoinedOutputs();
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                // the parts add their halves of every sum in another order than one network, so the last float digit may differ
                Assert.Equal((float)expected[i], (float)actual[i], 5);
            }
        }

        [Fact]
        public void Propagate_InAnyDeliveryOrder_GivesTheSameOutputs()
        {
            Wiring wiring = Wire(2);
            Quad[] inputs = { 0.5f, -1.25f, 0.75f };

            wiring.Propagate(inputs, 0, 1);
            Quad[] first = wiring.JoinedOutputs();
            wiring.Propagate(inputs, 1, 0);
            Quad[] second = wiring.JoinedOutputs();
            wiring.Propagate(inputs, 1, 0);
            Quad[] third = wiring.JoinedOutputs();

            for (int i = 0; i < first.Length; i++)
            {
                Assert.Equal((float)first[i], (float)second[i], 5);
                Assert.Equal((float)first[i], (float)third[i], 5);
            }
        }

        [Theory]
        [InlineData(2)]
        [InlineData(3)]
        public void Backpropagate_MatchesThePlainSectionGradientsAndWeightUpdates(int parts)
        {
            Wiring wiring = Wire(parts);
            Network plain = PlainSection();
            Dictionary<(int, int, int), BaseSynapse> plainSynapses = AssignPlainWeights(plain);
            Quad[] inputs = { 0.5f, -1.25f, 0.75f };
            Quad[] targets = { 0.9f, 0.1f, 0.4f };

            foreach (LearningRateOptimizer optimizer in plain.Optimizers.Concat(wiring.Parts.SelectMany(p => p.Optimizers)))
            {
                optimizer.LearningRate = 1f;
            }

            Quad[] values = plain.Propagate(inputs);
            plain.Backpropagate(targets);

            wiring.Propagate(inputs, Enumerable.Range(0, parts).ToArray());
            Quad[] outputGradient = values.Select((v, i) => v - targets[i]).ToArray();
            wiring.Backpropagate(outputGradient);

            Quad[] expectedInputGradient = plain.InputGradients;
            Quad[] actualInputGradient = wiring.SummedInputGradients();
            for (int i = 0; i < expectedInputGradient.Length; i++)
            {
                Assert.Equal((float)expectedInputGradient[i], (float)actualInputGradient[i], 5);
            }

            foreach (PartSynapse synapse in wiring.Parts.SelectMany(p => p.Synapses))
            {
                BaseSynapse expected = plainSynapses[(synapse.Layer, synapse.From, synapse.To)];
                Assert.NotEqual(Weight(synapse.Layer, synapse.From, synapse.To), (float)expected.Weight);
                Assert.Equal((float)expected.Weight, (float)synapse.Synapse.Weight, 5);
            }
        }

        [Fact]
        public void Partition_DescribesSlicesAndInputs()
        {
            NetworkPartition partition = Partition(2);

            Assert.Equal(1, partition.FirstLayer);
            Assert.Equal(3, partition.LastLayer);
            Assert.Equal(3, partition.LayerCount);
            Assert.Equal(INPUTS, partition.InputWidth);
            Assert.True(partition.InputHasBias);
            Assert.Equal(3, partition.OutputWidth);
            Assert.Equal(0, partition.SliceStart(1, 0));
            Assert.Equal(2, partition.SliceCount(1, 0));
            Assert.Equal(2, partition.SliceStart(1, 1));
            Assert.Equal(3, partition.SliceCount(1, 1));
            Assert.Equal(1, partition.PartOf(1, 4));

            NetworkSettings whole = SplitHelpers.Whole(2, 2);
            Assert.Throws<ArgumentOutOfRangeException>(() => RemoteNetworkFactory.Partition(whole, LAYERS, 0, 3, 2));
            Assert.Throws<ArgumentOutOfRangeException>(() => RemoteNetworkFactory.Partition(whole, LAYERS, 2, 1, 2));
            Assert.Throws<ArgumentException>(() => RemoteNetworkFactory.Partition(whole, LAYERS, 0, 2, 4));
        }
    }
}
