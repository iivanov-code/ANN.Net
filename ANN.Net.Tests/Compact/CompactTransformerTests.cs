using System;
using System.IO;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using Xunit;

namespace ANN.Net.Tests.Compact
{
    public class CompactTransformerTests
    {
        private static ISequenceNetwork Trained(FeedForwardType feedForward, bool causal, bool softMax)
        {
            ISequenceNetwork network = NetworkFactory.BuildTransformer(new TransformerSettings
            {
                InputDimension = 6, ModelDimension = 16, HeadsCount = 4, FeedForwardDimension = 24, LayersCount = 2, OutputDimension = 6,
                MaxSequenceLength = 8, UseCausalMask = causal, OutputApplySoftMax = softMax, FeedForwardType = feedForward,
                FeedForwardActivationType = ActivationTypes.GELU, OptimizerType = OptimizerTypes.Adam, Seed = 3
            });
            foreach (LearningRateOptimizer optimizer in network.Optimizers) optimizer.LearningRate = 0.01f;

            // gains, offsets and biases move away from their initial 1 and 0
            for (int step = 0; step < 15; step++)
            {
                network.Propagate(OneHot(0, 1, 2, 3, 4, 5));
                network.Backpropagate(OneHot(1, 2, 3, 4, 5, 0));
            }

            return network;
        }

        private static Quad[][] OneHot(params int[] tokens) =>
            tokens.Select(k => Enumerable.Range(0, 6).Select(i => (Quad)(i == k ? 1 : 0)).ToArray()).ToArray();

        private static Quad[][] Random(int length, int seed)
        {
            var random = new Random(seed);
            return Enumerable.Range(0, length).Select(_ => Enumerable.Range(0, 6).Select(_ => (Quad)((random.NextDouble() * 2) - 1)).ToArray()).ToArray();
        }

        private static float Largest(Quad[][] a, Quad[][] b) => a.Zip(b, (x, y) => x.Zip(y, (p, q) => Math.Abs(p - q)).Max()).Max();

        [Theory]
        [InlineData(true, true)]
        [InlineData(false, false)]
        public void MatrixTransformer_ComputesWhatTheTransformerComputes(bool causal, bool softMax)
        {
            ISequenceNetwork network = Trained(FeedForwardType.Matrix, causal, softMax);
            ICompactSequenceNetwork compact = NetworkCompactor.CompactSequence(network);

            Assert.Equal(network.Optimizers.Cast<ANN.Net.Optimizers.ArrayOptimizer>().Sum(x => (long)x.Length), compact.ParameterCount);
            foreach (int length in new[] { 1, 5, 8 })
            {
                Quad[][] sequence = Random(length, length);
                Assert.True(Largest(network.Propagate(sequence), compact.Propagate(sequence)) <= 1e-6f);
            }
        }

        [Fact]
        public void NeuronFeedForwardTransformer_ComputesWhatTheTransformerComputes()
        {
            ISequenceNetwork network = Trained(FeedForwardType.NeuronGraph, true, true);
            ICompactSequenceNetwork compact = NetworkCompactor.CompactSequence(network);

            Quad[][] sequence = Random(7, 1);
            Assert.True(Largest(network.Propagate(sequence), compact.Propagate(sequence)) <= 1e-5f);
        }

        [Theory]
        [InlineData(WeightPrecision.Half, 2e-3)]
        [InlineData(WeightPrecision.Int8, 3e-2)]
        public void SmallerWeights_StayClose(WeightPrecision precision, double tolerance)
        {
            ISequenceNetwork network = Trained(FeedForwardType.Matrix, true, true);
            ICompactSequenceNetwork single = NetworkCompactor.CompactSequence(network);
            ICompactSequenceNetwork compact = NetworkCompactor.CompactSequence(network, new CompactOptions { Precision = precision });

            Assert.True(compact.SizeInBytes < single.SizeInBytes * (precision == WeightPrecision.Half ? 0.6 : 0.4));
            Quad[][] sequence = Random(8, 2);
            Assert.True(Largest(network.Propagate(sequence), compact.Propagate(sequence)) <= tolerance);
        }

        [Fact]
        public void PropagateBatch_GivesTheOutputsOfEverySequence()
        {
            ICompactSequenceNetwork compact = NetworkCompactor.CompactSequence(Trained(FeedForwardType.Matrix, true, true), new CompactOptions { Precision = WeightPrecision.Int8 });
            Quad[][][] sequences = Enumerable.Range(1, 8).Select(length => Random(length, 10 + length)).ToArray();

            float[][][] batch = compact.PropagateBatch(sequences.Select(s => s.Select(v => v.Select(x => (float)x).ToArray()).ToArray()).ToArray());
            for (int i = 0; i < sequences.Length; i++)
            {
                Quad[][] single = compact.Propagate(sequences[i]);
                Assert.Equal(single.Select(v => v.Select(x => (float)x)), batch[i]);
            }
        }

        [Theory]
        [InlineData(FeedForwardType.Matrix, WeightPrecision.Single)]
        [InlineData(FeedForwardType.NeuronGraph, WeightPrecision.Int8)]
        public void SaveAndLoad_GivesTheSameOutputs(FeedForwardType feedForward, WeightPrecision precision)
        {
            ICompactSequenceNetwork compact = NetworkCompactor.CompactSequence(Trained(feedForward, true, true), new CompactOptions { Precision = precision, PruneFraction = 0.3 });
            using var stream = new MemoryStream();
            CompactNetworkSerializer.Save(compact, stream);

            ICompactSequenceNetwork loaded = CompactNetworkSerializer.LoadSequence(new MemoryStream(stream.ToArray()));
            Assert.Equal(compact.ParameterCount, loaded.ParameterCount);
            Assert.Equal(compact.SizeInBytes, loaded.SizeInBytes);
            Quad[][] sequence = Random(6, 3);
            Assert.Equal(compact.Propagate(sequence).Select(v => v.Select(x => (float)x)), loaded.Propagate(sequence).Select(v => v.Select(x => (float)x)));

            // the wrong kind of file, and damaged files
            Assert.Throws<InvalidDataException>(() => CompactNetworkSerializer.Load(new MemoryStream(stream.ToArray())));
            byte[] file = stream.ToArray();
            var random = new Random(4);
            for (int i = 0; i < 200; i++)
            {
                byte[] damaged = random.Next(2) == 0 ? file.Take(random.Next(file.Length)).ToArray() : (byte[])file.Clone();
                if (damaged.Length == file.Length) damaged[random.Next(damaged.Length)] = (byte)random.Next(256);
                try
                {
                    CompactNetworkSerializer.LoadSequence(new MemoryStream(damaged)).Propagate(Random(4, 5));
                }
                catch (InvalidDataException)
                {
                    // rejected
                }
            }
        }

        [Fact]
        public void Propagate_RejectsWrongSequences()
        {
            ICompactSequenceNetwork compact = NetworkCompactor.CompactSequence(Trained(FeedForwardType.Matrix, false, false));
            Assert.Throws<ArgumentException>(() => compact.Propagate(Array.Empty<Quad[]>()));
            Assert.Throws<ArgumentException>(() => compact.Propagate(Random(9, 1)));
            Assert.Throws<ArgumentException>(() => compact.Propagate(new[] { new Quad[5] }));
        }
    }
}
