using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Transformers;
using Xunit;

namespace ANN.Net.Tests.Transformers
{
    public class GradientCheckTests
    {
        private const int LENGTH = 4;
        private const int DIMENSION = 6;

        [Fact]
        public void LinearProjection_InputGradientsMatchFiniteDifferences()
        {
            var random = new Random(1);
            var projection = new LinearProjection(DIMENSION, 5, OptimizerTypes.Normal, WeightInitType.Xavier, new List<LearningRateOptimizer>());

            GradientChecker.AssertInputGradients(projection.Propagate,
                                                 projection.Backpropagate,
                                                 GradientChecker.RandomSequence(random, LENGTH, DIMENSION),
                                                 GradientChecker.RandomSequence(random, LENGTH, 5));
        }

        [Fact]
        public void NeuronFeedForwardNetwork_InputGradientsMatchFiniteDifferences()
        {
            var random = new Random(7);
            var network = new NeuronFeedForwardNetwork(DIMENSION, 5, ActivationTypes.HyperbolicTangens, OptimizerTypes.Normal, WeightInitType.Xavier, new List<LearningRateOptimizer>());

            GradientChecker.AssertInputGradients(network.Propagate,
                                                 network.Backpropagate,
                                                 GradientChecker.RandomSequence(random, LENGTH, DIMENSION),
                                                 GradientChecker.RandomSequence(random, LENGTH, DIMENSION));
        }

        [Fact]
        public void LinearProjection_WeightAndBiasGradientsMatchFiniteDifferences()
        {
            var random = new Random(2);
            var optimizers = new List<LearningRateOptimizer>();
            var projection = new LinearProjection(3, 2, OptimizerTypes.Normal, WeightInitType.Xavier, optimizers);
            Quad[][] inputs = GradientChecker.RandomSequence(random, LENGTH, 3);
            Quad[][] upstream = GradientChecker.RandomSequence(random, LENGTH, 2);

            double[,] numericWeights = new double[3, 2];
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    Quad original = projection.Weights[i, j];
                    projection.Weights[i, j] = original + (Quad)GradientChecker.EPSILON;
                    double plus = GradientChecker.Loss(projection.Propagate(inputs), upstream);
                    projection.Weights[i, j] = original - (Quad)GradientChecker.EPSILON;
                    double minus = GradientChecker.Loss(projection.Propagate(inputs), upstream);
                    projection.Weights[i, j] = original;
                    numericWeights[i, j] = (plus - minus) / (2 * GradientChecker.EPSILON);
                }
            }

            double[] numericBiases = new double[2];
            for (int j = 0; j < 2; j++)
            {
                Quad original = projection.Biases[j];
                projection.Biases[j] = original + (Quad)GradientChecker.EPSILON;
                double plus = GradientChecker.Loss(projection.Propagate(inputs), upstream);
                projection.Biases[j] = original - (Quad)GradientChecker.EPSILON;
                double minus = GradientChecker.Loss(projection.Propagate(inputs), upstream);
                projection.Biases[j] = original;
                numericBiases[j] = (plus - minus) / (2 * GradientChecker.EPSILON);
            }

            // with learning rate 1 the plain optimizer subtracts exactly the gradient
            Quad[,] weightsBefore = (Quad[,])projection.Weights.Clone();
            Quad[] biasesBefore = (Quad[])projection.Biases.Clone();
            foreach (LearningRateOptimizer optimizer in optimizers)
            {
                optimizer.LearningRate = 1f;
            }

            projection.Propagate(inputs);
            projection.Backpropagate(upstream);

            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    GradientChecker.AssertClose(numericWeights[i, j], weightsBefore[i, j] - projection.Weights[i, j], $"weight gradient [{i},{j}]");
                }
            }

            for (int j = 0; j < 2; j++)
            {
                GradientChecker.AssertClose(numericBiases[j], biasesBefore[j] - projection.Biases[j], $"bias gradient [{j}]");
            }
        }

        [Fact]
        public void LayerNormalization_InputGradientsMatchFiniteDifferences()
        {
            var random = new Random(3);
            var normalization = new LayerNormalization(DIMENSION, OptimizerTypes.Normal, new List<LearningRateOptimizer>());
            for (int i = 0; i < DIMENSION; i++)
            {
                normalization.Gains[i] = (Quad)(0.5 + random.NextDouble());
                normalization.Offsets[i] = (Quad)(random.NextDouble() - 0.5);
            }

            GradientChecker.AssertInputGradients(normalization.Propagate,
                                                 normalization.Backpropagate,
                                                 GradientChecker.RandomSequence(random, LENGTH, DIMENSION, 3),
                                                 GradientChecker.RandomSequence(random, LENGTH, DIMENSION));
        }

        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public void MultiHeadAttention_InputGradientsMatchFiniteDifferences(bool useCausalMask)
        {
            var random = new Random(4);
            var attention = new MultiHeadAttention(DIMENSION, 2, useCausalMask, OptimizerTypes.Normal, WeightInitType.Xavier, new List<LearningRateOptimizer>());

            GradientChecker.AssertInputGradients(attention.Propagate,
                                                 attention.Backpropagate,
                                                 GradientChecker.RandomSequence(random, LENGTH, DIMENSION),
                                                 GradientChecker.RandomSequence(random, LENGTH, DIMENSION));
        }

        [Fact]
        public void FeedForwardNetwork_InputGradientsMatchFiniteDifferences()
        {
            var random = new Random(5);
            var feedForward = new FeedForwardNetwork(DIMENSION, 10, ActivationTypes.ArcusTangens, OptimizerTypes.Normal, WeightInitType.Xavier, new List<LearningRateOptimizer>());

            GradientChecker.AssertInputGradients(feedForward.Propagate,
                                                 feedForward.Backpropagate,
                                                 GradientChecker.RandomSequence(random, LENGTH, DIMENSION),
                                                 GradientChecker.RandomSequence(random, LENGTH, DIMENSION));
        }

        [Theory]
        [InlineData(false, FeedForwardType.Matrix)]
        [InlineData(true, FeedForwardType.Matrix)]
        [InlineData(false, FeedForwardType.NeuronGraph)]
        [InlineData(true, FeedForwardType.NeuronGraph)]
        public void TransformerEncoderBlock_InputGradientsMatchFiniteDifferences(bool useCausalMask, FeedForwardType feedForwardType)
        {
            var random = new Random(6);
            var block = new TransformerEncoderBlock(DIMENSION, 3, 8, useCausalMask, ActivationTypes.ArcusTangens, feedForwardType, OptimizerTypes.Normal, WeightInitType.Xavier, new List<LearningRateOptimizer>());

            GradientChecker.AssertInputGradients(block.Propagate,
                                                 block.Backpropagate,
                                                 GradientChecker.RandomSequence(random, LENGTH, DIMENSION),
                                                 GradientChecker.RandomSequence(random, LENGTH, DIMENSION));
        }
    }
}
