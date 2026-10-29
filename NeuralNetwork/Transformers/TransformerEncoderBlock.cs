using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Optimizers;

namespace ANN.Net.Transformers
{
    /// <summary>
    /// One encoder block in the pre-normalization layout:
    /// x + Attention(Norm(x)) followed by h + FeedForward(Norm(h))
    /// </summary>
    [Serializable]
    internal class TransformerEncoderBlock
    {
        private readonly LayerNormalization attentionNormalization;
        private readonly MultiHeadAttention attention;
        private readonly LayerNormalization feedForwardNormalization;
        private readonly IFeedForwardNetwork feedForward;

        public TransformerEncoderBlock(int modelDimension,
                                       int headsCount,
                                       int feedForwardDimension,
                                       bool useCausalMask,
                                       ActivationTypes feedForwardActivationType,
                                       FeedForwardType feedForwardType,
                                       OptimizerTypes optimizerType,
                                       WeightInitType initializationType,
                                       ICollection<LearningRateOptimizer> optimizers)
        {
            this.attentionNormalization = new LayerNormalization(modelDimension, optimizerType, optimizers);
            this.attention = new MultiHeadAttention(modelDimension, headsCount, useCausalMask, optimizerType, initializationType, optimizers);
            this.feedForwardNormalization = new LayerNormalization(modelDimension, optimizerType, optimizers);
            this.feedForward = CreateFeedForward(feedForwardType, modelDimension, feedForwardDimension, feedForwardActivationType, optimizerType, initializationType, optimizers);
        }

        public MultiHeadAttention Attention => this.attention;

        public IFeedForwardNetwork FeedForward => this.feedForward;

        public Quad[][] Propagate(Quad[][] inputs)
        {
            Quad[][] attended = this.attention.Propagate(this.attentionNormalization.Propagate(inputs));
            Quad[][] residual = TransformerUtils.SequenceSum(inputs, attended);

            Quad[][] transformed = this.feedForward.Propagate(this.feedForwardNormalization.Propagate(residual));
            return TransformerUtils.SequenceSum(residual, transformed);
        }

        /// <summary>
        /// Propagates the gradient of the outputs back to the inputs and updates every sub-layer
        /// </summary>
        public Quad[][] Backpropagate(Quad[][] outputGradients)
        {
            Quad[][] feedForwardGradients = this.feedForwardNormalization.Backpropagate(this.feedForward.Backpropagate(outputGradients));
            Quad[][] residualGradients = TransformerUtils.SequenceSum(outputGradients, feedForwardGradients);

            Quad[][] attentionGradients = this.attentionNormalization.Backpropagate(this.attention.Backpropagate(residualGradients));
            return TransformerUtils.SequenceSum(residualGradients, attentionGradients);
        }

        private static IFeedForwardNetwork CreateFeedForward(FeedForwardType feedForwardType,
                                                             int modelDimension,
                                                             int feedForwardDimension,
                                                             ActivationTypes activationType,
                                                             OptimizerTypes optimizerType,
                                                             WeightInitType initializationType,
                                                             ICollection<LearningRateOptimizer> optimizers)
        {
            switch (feedForwardType)
            {
                case FeedForwardType.Matrix:
                    return new FeedForwardNetwork(modelDimension, feedForwardDimension, activationType, optimizerType, initializationType, optimizers);

                case FeedForwardType.NeuronGraph:
                    return new NeuronFeedForwardNetwork(modelDimension, feedForwardDimension, activationType, optimizerType, initializationType, optimizers);

                default:
                    throw new ArgumentException("Invalid value", nameof(feedForwardType));
            }
        }
    }
}
