using System;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Transformers;
using Xunit;

namespace ANN.Net.Tests.Transformers
{
    public class TransformerNetworkTests
    {
        private static TransformerSettings CreateSettings(bool useCausalMask = false, bool applySoftMax = false)
        {
            return new TransformerSettings
            {
                InputDimension = 4,
                ModelDimension = 8,
                HeadsCount = 2,
                FeedForwardDimension = 16,
                LayersCount = 2,
                OutputDimension = 4,
                MaxSequenceLength = 6,
                UseCausalMask = useCausalMask,
                OutputApplySoftMax = applySoftMax,
                OptimizerType = OptimizerTypes.Normal,
                InitializationType = WeightInitType.Xavier
            };
        }

        private static Quad[][] OneHotSequence(params int[] tokens)
        {
            Quad[][] sequence = new Quad[tokens.Length][];
            for (int t = 0; t < tokens.Length; t++)
            {
                sequence[t] = new Quad[4];
                sequence[t][tokens[t]] = 1;
            }

            return sequence;
        }

        private static int ArgMax(Quad[] values)
        {
            int best = 0;
            for (int i = 1; i < values.Length; i++)
            {
                if (values[i] > values[best])
                {
                    best = i;
                }
            }

            return best;
        }

        [Fact]
        public void Propagate_ReturnsOneOutputVectorPerInputVector()
        {
            ISequenceNetwork network = NetworkFactory.BuildTransformer(CreateSettings());

            Quad[][] outputs = network.Propagate(OneHotSequence(0, 1, 2));

            Assert.Equal(3, outputs.Length);
            Assert.All(outputs, output => Assert.Equal(4, output.Length));
            Assert.Same(outputs, network.Values);
        }

        [Fact]
        public void Propagate_WithSoftMax_EveryOutputVectorSumsToOne()
        {
            ISequenceNetwork network = NetworkFactory.BuildTransformer(CreateSettings(applySoftMax: true));

            Quad[][] outputs = network.Propagate(OneHotSequence(3, 2, 1, 0));

            foreach (Quad[] output in outputs)
            {
                double sum = 0;
                foreach (Quad value in output)
                {
                    Assert.True(value >= 0 && value <= 1);
                    sum += value;
                }

                Assert.Equal(1, sum, 4);
            }
        }

        [Fact]
        public void Propagate_SequenceLongerThanMaximum_Throws()
        {
            ISequenceNetwork network = NetworkFactory.BuildTransformer(CreateSettings());

            Assert.Throws<ArgumentException>(() => network.Propagate(OneHotSequence(0, 1, 2, 3, 0, 1, 2)));
        }

        [Fact]
        public void Backpropagate_WithoutPropagate_Throws()
        {
            ISequenceNetwork network = NetworkFactory.BuildTransformer(CreateSettings());

            Assert.Throws<InvalidOperationException>(() => network.Backpropagate(OneHotSequence(0)));
        }

        [Fact]
        public void Build_ModelDimensionNotDivisibleByHeads_Throws()
        {
            TransformerSettings settings = CreateSettings();
            settings.HeadsCount = 3;

            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildTransformer(settings));
        }

        [Fact]
        public void Propagate_WithCausalMask_LaterInputsDoNotChangeEarlierOutputs()
        {
            ISequenceNetwork network = NetworkFactory.BuildTransformer(CreateSettings(useCausalMask: true));

            Quad[][] first = network.Propagate(OneHotSequence(0, 1, 2, 3));
            Quad[][] second = network.Propagate(OneHotSequence(0, 1, 0, 0));

            for (int t = 0; t < 2; t++)
            {
                for (int i = 0; i < 4; i++)
                {
                    Assert.Equal((float)first[t][i], (float)second[t][i], 5);
                }
            }
        }

        [Fact]
        public void Propagate_WithoutCausalMask_LaterInputsChangeEarlierOutputs()
        {
            ISequenceNetwork network = NetworkFactory.BuildTransformer(CreateSettings(useCausalMask: false));

            Quad[][] first = network.Propagate(OneHotSequence(0, 1, 2, 3));
            Quad[][] second = network.Propagate(OneHotSequence(0, 1, 0, 0));

            bool anyDifferent = false;
            for (int i = 0; i < 4; i++)
            {
                anyDifferent |= Math.Abs(first[0][i] - second[0][i]) > 1e-6;
            }

            Assert.True(anyDifferent);
        }

        [Fact]
        public void Backpropagate_StoresErrorsAsPredictedMinusTarget()
        {
            ISequenceNetwork network = NetworkFactory.BuildTransformer(CreateSettings());
            Quad[][] targets = OneHotSequence(1, 2);

            Quad[][] outputs = network.Propagate(OneHotSequence(0, 1));
            Quad[][] predicted = new Quad[][] { (Quad[])outputs[0].Clone(), (Quad[])outputs[1].Clone() };
            network.Backpropagate(targets);

            for (int t = 0; t < 2; t++)
            {
                for (int i = 0; i < 4; i++)
                {
                    Assert.Equal((float)(predicted[t][i] - targets[t][i]), (float)network.Errors[t][i], 6);
                }
            }
        }

        [Fact]
        public void Backpropagate_RepeatedTraining_LearnsNextTokenPrediction()
        {
            ISequenceNetwork network = NetworkFactory.BuildTransformer(CreateSettings(useCausalMask: true, applySoftMax: true));
            foreach (LearningRateOptimizer optimizer in network.Optimizers)
            {
                optimizer.LearningRate = 0.05f;
            }

            Quad[][] inputs = OneHotSequence(0, 1, 2, 3, 0);
            Quad[][] targets = OneHotSequence(1, 2, 3, 0, 1);

            network.Propagate(inputs);
            Quad initialError = network.Backpropagate(targets);

            Quad finalError = initialError;
            for (int epoch = 0; epoch < 300; epoch++)
            {
                network.Propagate(inputs);
                finalError = network.Backpropagate(targets);
            }

            Assert.True(finalError < initialError / 4, $"error went from {initialError} to {finalError}");

            Quad[][] outputs = network.Propagate(inputs);
            for (int t = 0; t < inputs.Length; t++)
            {
                Assert.Equal(ArgMax(targets[t]), ArgMax(outputs[t]));
            }
        }

        [Fact]
        public void PositionalEncoding_GivesDistinctVectorsPerPosition()
        {
            Quad[][] encodings = PositionalEncoding.Build(6, 8);

            Assert.Equal(6, encodings.Length);
            Assert.Equal(0f, (float)encodings[0][0], 6);
            Assert.Equal(1f, (float)encodings[0][1], 6);

            for (int a = 0; a < encodings.Length; a++)
            {
                for (int b = a + 1; b < encodings.Length; b++)
                {
                    double distance = 0;
                    for (int i = 0; i < 8; i++)
                    {
                        distance += Math.Pow(encodings[a][i] - encodings[b][i], 2);
                    }

                    Assert.True(distance > 1e-3, $"positions {a} and {b} have the same encoding");
                }
            }
        }
    }
}
