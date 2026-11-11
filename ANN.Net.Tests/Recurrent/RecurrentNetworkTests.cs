using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using Xunit;

namespace ANN.Net.Tests.Recurrent
{
    public class RecurrentNetworkTests
    {
        private static IRecurrentNetwork Build(NeuronType cell, int seed = 3, OptimizerTypes optimizer = OptimizerTypes.Adam, int hidden = 6,
                                               int layers = 2, float clipping = 0, bool softMax = false) =>
            NetworkFactory.BuildRecurrent(new RecurrentNetworkSettings
            {
                InputDimension = 3, HiddenDimension = (ushort)hidden, LayersCount = (ushort)layers, OutputDimension = 2, CellType = cell,
                OptimizerType = optimizer, Seed = seed, GradientClipping = clipping, OutputApplySoftMax = softMax
            });

        private static Quad[][] Sequence(int length, int seed)
        {
            var random = new Random(seed);
            return Enumerable.Range(0, length).Select(_ => Enumerable.Range(0, 3).Select(_ => (Quad)((random.NextDouble() * 2) - 1)).ToArray()).ToArray();
        }

        private static Quad[][] Targets(int length, int seed)
        {
            var random = new Random(seed + 1000);
            return Enumerable.Range(0, length).Select(_ => new[] { (Quad)random.NextDouble(), (Quad)random.NextDouble() }).ToArray();
        }

        private static int[] Bits(IEnumerable<Quad> values) => values.Select(x => BitConverter.SingleToInt32Bits(x)).ToArray();

        private static void Rate(IEnumerable<LearningRateOptimizer> optimizers, float rate)
        {
            foreach (LearningRateOptimizer optimizer in optimizers) optimizer.LearningRate = rate;
        }

        [Theory]
        [InlineData(NeuronType.LSTM, (((3 + 6) * 24) + 24) + (((6 + 6) * 24) + 24) + 14)]
        [InlineData(NeuronType.LSTMPeepHole, (((3 + 6) * 24) + 24 + 18) + (((6 + 6) * 24) + 24 + 18) + 14)]
        [InlineData(NeuronType.GRU, (((3 + 6) * 18) + 18) + (((6 + 6) * 18) + 18) + 14)]
        [InlineData(NeuronType.RNN, (((3 + 6) * 6) + 6) + (((6 + 6) * 6) + 6) + 14)]
        public void ParameterCount_CountsEveryWeightAndBias(NeuronType cell, long expected)
        {
            IRecurrentNetwork network = Build(cell);

            Assert.Equal(expected, network.ParameterCount);
            Assert.Equal(cell, network.CellType);
            Assert.Equal(6, network.HiddenDimension);
            Assert.Equal(2, network.LayersCount);
        }

        [Theory]
        [InlineData(NeuronType.LSTM)]
        [InlineData(NeuronType.LSTMPeepHole)]
        [InlineData(NeuronType.GRU)]
        [InlineData(NeuronType.RNN)]
        public void Step_GivesWhatPropagateGivesAtTheSamePosition(NeuronType cell)
        {
            IRecurrentNetwork network = Build(cell, softMax: true);
            Quad[][] sequence = Sequence(7, 1);
            Quad[][] whole = network.Propagate(sequence);

            for (int pass = 0; pass < 2; pass++)
            {
                network.ResetState();
                for (int t = 0; t < sequence.Length; t++)
                {
                    Assert.Equal(Bits(whole[t]), Bits(network.Step(sequence[t])));
                }
            }
        }

        [Fact]
        public void EverySequenceStartsFromZeroState()
        {
            IRecurrentNetwork network = Build(NeuronType.LSTM);
            Quad[][] first = network.Propagate(Sequence(5, 2));
            network.Propagate(Sequence(9, 3));

            Assert.Equal(Bits(first.SelectMany(x => x)), Bits(network.Propagate(Sequence(5, 2)).SelectMany(x => x)));
        }

        [Fact]
        public void TheOutputDependsOnEarlierInputs()
        {
            IRecurrentNetwork network = Build(NeuronType.GRU);
            Quad[][] a = Sequence(4, 5), b = Sequence(4, 5);
            b[0] = new Quad[] { 1, -1, 1 };

            Assert.NotEqual(Bits(network.Propagate(a)[3]), Bits(network.Propagate(b)[3]));
        }

        [Fact]
        public void TheSameSeedBuildsTheSameNetwork()
        {
            Quad[][] sequence = Sequence(6, 4);
            Assert.Equal(Bits(Build(NeuronType.LSTM, seed: 9).Propagate(sequence).SelectMany(x => x)),
                         Bits(Build(NeuronType.LSTM, seed: 9).Propagate(sequence).SelectMany(x => x)));
            Assert.NotEqual(Bits(Build(NeuronType.LSTM, seed: 9).Propagate(sequence).SelectMany(x => x)),
                            Bits(Build(NeuronType.LSTM, seed: 10).Propagate(sequence).SelectMany(x => x)));
        }

        [Theory]
        [InlineData(NeuronType.LSTM, 4)]
        [InlineData(NeuronType.LSTMPeepHole, 4)]
        [InlineData(NeuronType.GRU, 3)]
        public void GatesThatKeepTheStateStartOpen(NeuronType cell, int gates)
        {
            // the forget gates of an LSTM and the update gates of a GRU are the second block of columns
            var network = (ANN.Net.Recurrent.RecurrentNetwork)Build(cell);
            foreach (ANN.Net.Recurrent.RecurrentLayer layer in network.Layers)
            {
                float[] biases = layer.Biases.Weights.ToArray();
                Assert.Equal(gates * 6, biases.Length);
                Assert.All(biases.Skip(6).Take(6), x => Assert.Equal(1f, x));
                Assert.All(biases.Take(6).Concat(biases.Skip(12)), x => Assert.Equal(0f, x));
            }
        }

        [Theory]
        [InlineData(NeuronType.LSTM)]
        [InlineData(NeuronType.GRU)]
        public void BatchesOfOne_TrainExactlyLikeOneSequenceAtATime(NeuronType cell)
        {
            IRecurrentNetwork online = Build(cell, clipping: 0.5f), batched = Build(cell, clipping: 0.5f);
            Rate(online.Optimizers.Concat(batched.Optimizers), 0.01f);

            for (int step = 0; step < 10; step++)
            {
                Quad[][] sequence = Sequence(3 + (step % 4), step), targets = Targets(3 + (step % 4), step);
                online.Propagate(sequence);
                Assert.Equal((float)online.Backpropagate(targets), (float)batched.TrainBatch(new[] { sequence }, new[] { targets }));
            }

            Assert.Equal(Bits(online.Propagate(Sequence(5, 99)).SelectMany(x => x)), Bits(batched.Propagate(Sequence(5, 99)).SelectMany(x => x)));
        }

        [Fact]
        public void TrainBatch_DoesNotDependOnTheNumberOfThreads()
        {
            IRecurrentNetwork one = Build(NeuronType.LSTMPeepHole), many = Build(NeuronType.LSTMPeepHole);
            Rate(one.Optimizers.Concat(many.Optimizers), 0.01f);
            var sequences = Enumerable.Range(0, 9).Select(i => Sequence(2 + (i % 5), i)).ToList();
            var targets = Enumerable.Range(0, 9).Select(i => Targets(2 + (i % 5), i)).ToList();

            for (int step = 0; step < 4; step++)
            {
                Assert.Equal((float)one.TrainBatch(sequences, targets, new BatchOptions { ChunkSize = 2, MaxDegreeOfParallelism = 1 }),
                             (float)many.TrainBatch(sequences, targets, new BatchOptions { ChunkSize = 2, MaxDegreeOfParallelism = -1 }));
            }

            Assert.Equal(Bits(one.Propagate(Sequence(5, 99)).SelectMany(x => x)), Bits(many.Propagate(Sequence(5, 99)).SelectMany(x => x)));
        }

        [Fact]
        public void GradientClipping_LimitsTheNormOfAStep()
        {
            var clipped = (ANN.Net.Recurrent.RecurrentNetwork)Build(NeuronType.LSTM, optimizer: OptimizerTypes.Normal, clipping: 0.01f);
            Rate(clipped.Optimizers, 1);
            List<Quad[]> before = clipped.Tensors.Select(x => (Quad[])x.Values.Clone()).ToList();

            clipped.Propagate(Sequence(6, 7));
            clipped.Backpropagate(Targets(6, 7));

            double squares = 0;
            for (int k = 0; k < before.Count; k++)
            {
                for (int i = 0; i < before[k].Length; i++)
                {
                    double change = before[k][i] - clipped.Tensors[k].Values[i];
                    squares += change * change;
                }
            }

            Assert.Equal(0.01, Math.Sqrt(squares), 4);
        }

        [Fact]
        public void TargetsMayBeLeftOutForStepsThatDoNotMatter()
        {
            IRecurrentNetwork network = Build(NeuronType.LSTM, softMax: true);
            Quad[][] outputs = network.Propagate(Sequence(4, 1));
            Quad[][] targets = { null, null, null, new Quad[] { 0, 1 } };

            Quad loss = network.Backpropagate(targets);

            Assert.Equal(-Math.Log(outputs[3][1]), loss, 4);
            Assert.All(network.Errors.Take(3), e => Assert.All(e, x => Assert.Equal(0f, (float)x)));
        }

        /// <summary>
        /// A bit comes at the first step, on the same input as noise that follows at every other step; a second input
        /// marks the step that counts. After 40 steps the network must tell which bit it was, so it has to keep the
        /// bit and ignore the noise
        /// </summary>
        [Theory]
        [InlineData(NeuronType.LSTM)]
        [InlineData(NeuronType.LSTMPeepHole)]
        [InlineData(NeuronType.GRU)]
        public void GatedCellsRememberABitAcrossFortyNoisySteps(NeuronType cell)
        {
            const int length = 40;
            IRecurrentNetwork network = NetworkFactory.BuildRecurrent(new RecurrentNetworkSettings
            {
                InputDimension = 2, HiddenDimension = 8, OutputDimension = 2, CellType = cell, OutputApplySoftMax = true,
                OptimizerType = OptimizerTypes.Adam, Seed = 1, GradientClipping = 1, ForgetGateBias = 3
            });
            Rate(network.Optimizers, 0.01f);

            var random = new Random(1);
            for (int batch = 0; batch < 200 && Accuracy(network, length) < 0.99; batch++)
            {
                var samples = Enumerable.Range(0, 16).Select(_ => Recall(random, length)).ToList();
                network.TrainBatch(samples.Select(x => x.Inputs).ToList(), samples.Select(x => x.Targets).ToList());
            }

            Assert.True(Accuracy(network, length) >= 0.99, $"{cell} recalls {Accuracy(network, length):P0} of the bits");
        }

        private static (Quad[][] Inputs, Quad[][] Targets) Recall(Random random, int length)
        {
            int bit = random.Next(2);
            Quad[][] inputs = Enumerable.Range(0, length)
                                        .Select(t => t == 0 ? new Quad[] { bit == 1 ? 1 : -1, 1 } : new Quad[] { (Quad)((random.NextDouble() * 2) - 1), 0 })
                                        .ToArray();
            var targets = new Quad[length][];
            targets[length - 1] = bit == 1 ? new Quad[] { 0, 1 } : new Quad[] { 1, 0 };
            return (inputs, targets);
        }

        private static double Accuracy(IRecurrentNetwork network, int length)
        {
            var random = new Random(999);
            int right = 0;
            for (int i = 0; i < 200; i++)
            {
                (Quad[][] inputs, Quad[][] targets) = Recall(random, length);
                Quad[] output = network.Propagate(inputs)[length - 1];
                if ((output[1] > output[0]) == (targets[length - 1][1] > 0.5f)) right++;
            }

            return right / 200.0;
        }

        [Fact]
        public void InvalidSettingsAndInputsAreRejected()
        {
            Assert.Throws<ArgumentNullException>(() => NetworkFactory.BuildRecurrent(null));
            Assert.Throws<ArgumentException>(() => Build(NeuronType.Normal));
            Assert.Throws<ArgumentException>(() => Build(NeuronType.LSTM, hidden: 0));
            Assert.Throws<ArgumentException>(() => Build(NeuronType.LSTM, layers: 0));
            Assert.Throws<ArgumentException>(() => Build(NeuronType.LSTM, clipping: -1));
            Assert.Throws<ArgumentException>(() => Build(NeuronType.LSTM, clipping: float.NaN));

            IRecurrentNetwork network = Build(NeuronType.GRU);
            Assert.Throws<InvalidOperationException>(() => network.Backpropagate(Targets(3, 1)));
            Assert.Throws<ArgumentException>(() => network.Propagate(Array.Empty<Quad[]>()));
            Assert.Throws<ArgumentException>(() => network.Propagate(new[] { new Quad[] { 1, 2 } }));
            Assert.Throws<ArgumentException>(() => network.Step(new Quad[] { 1 }));

            network.Propagate(Sequence(3, 1));
            Assert.Throws<ArgumentException>(() => network.Backpropagate(Targets(2, 1)));
            Assert.Throws<ArgumentException>(() => network.Backpropagate(new[] { new Quad[] { 1 }, null, null }));
        }

        [Fact]
        public void CellLayersInNeuronNetworksPointToBuildRecurrent()
        {
            var hidden = new RecurrentLayerSettings
            {
                NeuronsCount = 4, LayersCount = 1, CellType = NeuronType.LSTM, PrevLayerConnectionType = LayerConnectionType.FullyConnected,
                ActivationType = ActivationTypes.HyperbolicTangens
            };
            var settings = new NetworkSettings
            {
                Seed = 1, IdentityGenerator = new IdentityGenerators.IncrementalGenerator(),
                InputSettings = new InputLayerSettings { NeuronsCount = 3, ActivationType = ActivationTypes.Identity },
                OutputSettings = new OutputLayerSettings { NeuronsCount = 1, ActivationType = ActivationTypes.Sigmoid }
            };

            foreach (NeuronType cell in new[] { NeuronType.LSTM, NeuronType.LSTMPeepHole, NeuronType.GRU })
            {
                hidden.CellType = cell;
                NotSupportedException error = Assert.Throws<NotSupportedException>(() => NetworkFactory.BuildFFN(settings, hidden));
                Assert.Contains(nameof(NetworkFactory.BuildRecurrent), error.Message, StringComparison.Ordinal);
            }
        }
    }
}
