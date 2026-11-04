using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.IdentityGenerators;
using Xunit;

namespace ANN.Net.Tests.Optimizers
{
    /// <summary>
    /// Mini-batches trained on several threads
    /// </summary>
    public class BatchTrainingTests
    {
        private static IMatrixNetwork Matrix(OptimizerTypes optimizer = OptimizerTypes.Adam, bool softMax = true) => NetworkFactory.BuildMatrixFFN(new NetworkSettings
        {
            Seed = 1,
            IdentityGenerator = new IncrementalGenerator(),
            LossType = LossFunctionTypes.Squared,
            ShouldNormalize = true,
            MinMaxValues = Enumerable.Range(0, 6).Select(i => Tuple.Create<Quad, Quad>(-2, 2 + i)).ToList(),
            OutputApplySoftMax = softMax,
            InputSettings = new InputLayerSettings { NeuronsCount = 6, HasBiasNeuron = true, ActivationType = ActivationTypes.SoftSign },
            OutputSettings = new OutputLayerSettings { NeuronsCount = 3, ActivationType = ActivationTypes.Sigmoid, OptimizerType = optimizer, InitializationType = WeightInitType.Xavier }
        },
        new RecurrentLayerSettings
        {
            NeuronsCount = 10, LayersCount = 2, CellType = NeuronType.Normal, PrevLayerConnectionType = LayerConnectionType.FullyConnected,
            ActivationType = ActivationTypes.HyperbolicTangens, HasBiasNeuron = true, OptimizerType = optimizer, InitializationType = WeightInitType.Xavier
        });

        private static IConvolutionalNetwork Convolutional() => NetworkFactory.BuildConvolutional(new ConvolutionalNetworkSettings
        {
            InputChannels = 2, InputHeight = 8, InputWidth = 8, OutputDimension = 3, OutputApplySoftMax = true, Seed = 2, OptimizerType = OptimizerTypes.Adam,
            Layers =
            {
                new ConvolutionLayerSettings { Filters = 4, KernelSize = 3, Padding = 1, UseBias = false, ActivationType = ActivationTypes.Identity },
                new BatchNormalizationLayerSettings { ActivationType = ActivationTypes.RELU },
                new ResidualBlockSettings { Filters = 6, Stride = 2 },
                new PoolingLayerSettings { Type = PoolingType.Max, Size = 2 },
                new GlobalAveragePoolingLayerSettings(),
                new DenseLayerSettings { NeuronsCount = 5, ActivationType = ActivationTypes.HyperbolicTangens }
            }
        });

        private static ISequenceNetwork Transformer(FeedForwardType feedForward = FeedForwardType.Matrix) => NetworkFactory.BuildTransformer(new TransformerSettings
        {
            InputDimension = 4, ModelDimension = 8, HeadsCount = 2, FeedForwardDimension = 12, LayersCount = 2, OutputDimension = 4,
            MaxSequenceLength = 6, UseCausalMask = true, OutputApplySoftMax = true, OptimizerType = OptimizerTypes.Adam, Seed = 3, FeedForwardType = feedForward
        });

        private static void Rate(IEnumerable<LearningRateOptimizer> optimizers, float rate)
        {
            foreach (LearningRateOptimizer optimizer in optimizers) optimizer.LearningRate = rate;
        }

        private static Quad[] Vector(int size, int seed)
        {
            var random = new Random(seed);
            return Enumerable.Range(0, size).Select(_ => (Quad)((random.NextDouble() * 2) - 1)).ToArray();
        }

        private static Quad[] OneHot(int size, int k) => Enumerable.Range(0, size).Select(i => (Quad)(i == k ? 1 : 0)).ToArray();

        private static Quad[][] Sequence(int length, int seed) => Enumerable.Range(0, length).Select(t => Vector(4, (seed * 10) + t)).ToArray();

        private static Quad[][] Targets(int length, int seed) => Enumerable.Range(0, length).Select(t => OneHot(4, (seed + t) % 4)).ToArray();

        private static int[] Bits(IEnumerable<Quad> values) => values.Select(x => BitConverter.SingleToInt32Bits(x)).ToArray();

        [Fact]
        public void BatchesOfOne_TrainExactlyLikeOneSampleAtATime()
        {
            IMatrixNetwork online = Matrix(), batched = Matrix();
            IConvolutionalNetwork onlineCnn = Convolutional(), batchedCnn = Convolutional();
            ISequenceNetwork onlineTransformer = Transformer(), batchedTransformer = Transformer();
            Rate(online.Optimizers.Concat(batched.Optimizers).Concat(onlineCnn.Optimizers).Concat(batchedCnn.Optimizers)
                       .Concat(onlineTransformer.Optimizers).Concat(batchedTransformer.Optimizers), 0.02f);

            for (int step = 0; step < 12; step++)
            {
                Quad[] x = Vector(6, step), t = OneHot(3, step % 3);
                online.Propagate(x);
                Assert.Equal((float)online.Backpropagate(t), (float)batched.TrainBatch(new[] { x }, new[] { t }));

                Quad[] image = Vector(128, step + 100);
                onlineCnn.Propagate(image);
                Assert.Equal((float)onlineCnn.Backpropagate(t), (float)batchedCnn.TrainBatch(new[] { image }, new[] { t }));

                onlineTransformer.Propagate(Sequence(5, step));
                Assert.Equal((float)onlineTransformer.Backpropagate(Targets(5, step)), (float)batchedTransformer.TrainBatch(new[] { Sequence(5, step) }, new[] { Targets(5, step) }));
            }

            Quad[] probe = Vector(6, 99);
            Assert.Equal(Bits(online.Propagate(probe)), Bits(batched.Propagate(probe)));

            // the running averages of batch normalization advance the same way
            onlineCnn.Training = batchedCnn.Training = false;
            Quad[] probeImage = Vector(128, 199);
            Assert.Equal(Bits(onlineCnn.Propagate(probeImage)), Bits(batchedCnn.Propagate(probeImage)));

            Assert.Equal(Bits(onlineTransformer.Propagate(Sequence(4, 99)).SelectMany(x => x)), Bits(batchedTransformer.Propagate(Sequence(4, 99)).SelectMany(x => x)));
        }

        [Theory]
        [InlineData(1, 3)]
        [InlineData(3, 4)]
        [InlineData(8, 1)]
        public void Results_DoNotDependOnTheNumberOfThreads(int chunkSize, int threads)
        {
            IMatrixNetwork reference = Matrix(), other = Matrix();
            IConvolutionalNetwork referenceCnn = Convolutional(), otherCnn = Convolutional();
            ISequenceNetwork referenceTransformer = Transformer(), otherTransformer = Transformer();
            Rate(reference.Optimizers.Concat(other.Optimizers).Concat(referenceCnn.Optimizers).Concat(otherCnn.Optimizers)
                          .Concat(referenceTransformer.Optimizers).Concat(otherTransformer.Optimizers), 0.02f);

            var one = new BatchOptions { ChunkSize = chunkSize, MaxDegreeOfParallelism = 1 };
            var many = new BatchOptions { ChunkSize = chunkSize, MaxDegreeOfParallelism = threads };
            for (int step = 0; step < 4; step++)
            {
                Quad[][] inputs = Enumerable.Range(0, 13).Select(i => Vector(6, (step * 100) + i)).ToArray();
                Quad[][] targets = Enumerable.Range(0, 13).Select(i => OneHot(3, i % 3)).ToArray();
                Assert.Equal((float)reference.TrainBatch(inputs, targets, one), (float)other.TrainBatch(inputs, targets, many));

                Quad[][] images = Enumerable.Range(0, 7).Select(i => Vector(128, (step * 100) + i)).ToArray();
                Assert.Equal((float)referenceCnn.TrainBatch(images, targets.Take(7).ToList(), one), (float)otherCnn.TrainBatch(images, targets.Take(7).ToList(), many));

                Quad[][][] sequences = Enumerable.Range(0, 6).Select(i => Sequence(2 + (i % 4), (step * 10) + i)).ToArray();
                Quad[][][] sequenceTargets = sequences.Select((s, i) => Targets(s.Length, i)).ToArray();
                Assert.Equal((float)referenceTransformer.TrainBatch(sequences, sequenceTargets, one), (float)otherTransformer.TrainBatch(sequences, sequenceTargets, many));
            }

            Quad[] probe = Vector(6, 5);
            Assert.Equal(Bits(reference.Propagate(probe)), Bits(other.Propagate(probe)));
            Assert.Equal(Bits(referenceCnn.Propagate(Vector(128, 6))), Bits(otherCnn.Propagate(Vector(128, 6))));
            Assert.Equal(Bits(referenceTransformer.Propagate(Sequence(3, 7)).SelectMany(x => x)), Bits(otherTransformer.Propagate(Sequence(3, 7)).SelectMany(x => x)));
        }

        [Fact]
        public void BatchStep_IsTheAverageOfTheStepsOfItsSamples()
        {
            // plain gradient descent: the batch moves every weight by the mean of the moves of its samples
            Quad[][] inputs = Enumerable.Range(0, 5).Select(i => Vector(6, 40 + i)).ToArray();
            Quad[][] targets = Enumerable.Range(0, 5).Select(i => OneHot(3, i % 3)).ToArray();

            var batched = (Matrix.MatrixNetwork)Matrix(OptimizerTypes.Normal, softMax: false);
            Rate(batched.Optimizers, 0.1f);
            float[] start = batched.Layers.SelectMany(l => l.Weights).Select(x => (float)x).ToArray();
            batched.TrainBatch(inputs, targets, new BatchOptions { ChunkSize = 2 });
            float[] after = batched.Layers.SelectMany(l => l.Weights).Select(x => (float)x).ToArray();

            var moves = new float[start.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                var single = (Matrix.MatrixNetwork)Matrix(OptimizerTypes.Normal, softMax: false);
                Rate(single.Optimizers, 0.1f);
                single.Propagate(inputs[i]);
                single.Backpropagate(targets[i]);
                float[] weights = single.Layers.SelectMany(l => l.Weights).Select(x => (float)x).ToArray();
                for (int k = 0; k < moves.Length; k++) moves[k] += (weights[k] - start[k]) / inputs.Length;
            }

            for (int k = 0; k < moves.Length; k++) Assert.Equal(start[k] + moves[k], after[k], 1e-6f);
        }

        [Fact]
        public void Batches_LearnAtLeastAsWellAsOneSampleAtATime()
        {
            Quad[][] inputs = Enumerable.Range(0, 96).Select(i => Vector(6, 500 + i)).ToArray();
            Quad[][] targets = inputs.Select(x => OneHot(3, x[0] > 0.3f ? 0 : x[1] > 0 ? 1 : 2)).ToArray();
            int Correct(IMatrixNetwork network) =>
                inputs.Zip(targets, (x, t) => { Quad[] y = network.Propagate(x); return Array.IndexOf(y, y.Max()) == Array.IndexOf(t, (Quad)1); }).Count(x => x);

            IMatrixNetwork online = Matrix(softMax: false), batched = Matrix(softMax: false);
            Rate(online.Optimizers.Concat(batched.Optimizers), 0.01f);
            Quad first = batched.TrainBatch(inputs, targets), last = first;
            for (int epoch = 0; epoch < 300; epoch++)
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    online.Propagate(inputs[i]);
                    online.Backpropagate(targets[i]);
                }

                for (int b = 0; b < inputs.Length; b += 32) last = batched.TrainBatch(inputs.Skip(b).Take(32).ToList(), targets.Skip(b).Take(32).ToList());
            }

            Assert.True(last < first / 2, $"loss {first} -> {last}");
            Assert.True(Correct(batched) >= Correct(online), $"batches {Correct(batched)}, one sample at a time {Correct(online)} of 96");
        }

        [Fact]
        public void TrainBatch_RejectsWhatItCannotTrain()
        {
            IMatrixNetwork network = Matrix();
            Assert.Throws<ArgumentException>(() => network.TrainBatch(Array.Empty<Quad[]>(), Array.Empty<Quad[]>()));
            Assert.Throws<ArgumentException>(() => network.TrainBatch(new[] { Vector(6, 1) }, Array.Empty<Quad[]>()));
            Assert.Throws<ArgumentOutOfRangeException>(() => network.TrainBatch(new[] { Vector(6, 1) }, new[] { OneHot(3, 0) }, new BatchOptions { ChunkSize = 0 }));
            Assert.Throws<NotSupportedException>(() => Transformer(FeedForwardType.NeuronGraph).TrainBatch(new[] { Sequence(2, 1) }, new[] { Targets(2, 1) }));
        }
    }
}
