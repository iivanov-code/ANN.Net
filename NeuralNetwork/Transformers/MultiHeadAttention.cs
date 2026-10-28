using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Optimizers;

namespace ANN.Net.Transformers
{
    /// <summary>
    /// Scaled dot-product self attention with several heads.
    /// Queries, keys and values are produced by three <see cref="LinearProjection"/>s,
    /// the concatenated heads are mixed by a fourth one
    /// </summary>
    [Serializable]
    internal class MultiHeadAttention
    {
        private readonly int headsCount;
        private readonly int headDimension;
        private readonly bool useCausalMask;
        private readonly Quad scale;
        private readonly LinearProjection queryProjection;
        private readonly LinearProjection keyProjection;
        private readonly LinearProjection valueProjection;
        private readonly LinearProjection outputProjection;

        private Quad[][] lastQueries;
        private Quad[][] lastKeys;
        private Quad[][] lastValues;
        private Quad[][][] lastAttentionWeights;

        public MultiHeadAttention(int modelDimension,
                                  int headsCount,
                                  bool useCausalMask,
                                  OptimizerTypes optimizerType,
                                  WeightInitType initializationType,
                                  ICollection<LearningRateOptimizer> optimizers)
        {
            if (modelDimension < 1) throw new ArgumentOutOfRangeException(nameof(modelDimension), "Must be greater than 0");
            if (headsCount < 1) throw new ArgumentOutOfRangeException(nameof(headsCount), "Must be greater than 0");
            if (modelDimension % headsCount != 0)
                throw new ArgumentException("Model dimension must be divisible by the heads count", nameof(headsCount));

            this.headsCount = headsCount;
            this.headDimension = modelDimension / headsCount;
            this.useCausalMask = useCausalMask;
            this.scale = (Quad)(1 / Math.Sqrt(this.headDimension));

            this.queryProjection = new LinearProjection(modelDimension, modelDimension, optimizerType, initializationType, optimizers);
            this.keyProjection = new LinearProjection(modelDimension, modelDimension, optimizerType, initializationType, optimizers);
            this.valueProjection = new LinearProjection(modelDimension, modelDimension, optimizerType, initializationType, optimizers);
            this.outputProjection = new LinearProjection(modelDimension, modelDimension, optimizerType, initializationType, optimizers);
        }

        public int ModelDimension => this.headsCount * this.headDimension;

        public int HeadsCount => this.headsCount;

        /// <summary>
        /// Attention weights of the last propagation indexed as [head][query position][key position]
        /// </summary>
        public Quad[][][] AttentionWeights => this.lastAttentionWeights;

        public Quad[][] Propagate(Quad[][] inputs)
        {
            if (inputs == null) throw new ArgumentNullException(nameof(inputs));

            int length = inputs.Length;
            Quad[][] queries = this.queryProjection.Propagate(inputs);
            Quad[][] keys = this.keyProjection.Propagate(inputs);
            Quad[][] values = this.valueProjection.Propagate(inputs);

            Quad[][] concatenated = TransformerUtils.CreateSequence(length, this.ModelDimension);
            Quad[][][] attentionWeights = new Quad[this.headsCount][][];

            for (int head = 0; head < this.headsCount; head++)
            {
                int offset = head * this.headDimension;
                attentionWeights[head] = new Quad[length][];

                for (int i = 0; i < length; i++)
                {
                    Quad[] scores = new Quad[length];
                    for (int j = 0; j < length; j++)
                    {
                        if (this.IsMasked(i, j))
                        {
                            scores[j] = Quad.NegativeInfinity;
                            continue;
                        }

                        Quad dot = 0;
                        for (int c = 0; c < this.headDimension; c++)
                        {
                            dot += queries[i][offset + c] * keys[j][offset + c];
                        }

                        scores[j] = dot * this.scale;
                    }

                    TransformerUtils.SoftMax(scores);
                    attentionWeights[head][i] = scores;

                    for (int c = 0; c < this.headDimension; c++)
                    {
                        Quad sum = 0;
                        for (int j = 0; j < length; j++)
                        {
                            sum += scores[j] * values[j][offset + c];
                        }

                        concatenated[i][offset + c] = sum;
                    }
                }
            }

            this.lastQueries = queries;
            this.lastKeys = keys;
            this.lastValues = values;
            this.lastAttentionWeights = attentionWeights;

            return this.outputProjection.Propagate(concatenated);
        }

        /// <summary>
        /// Propagates the gradient of the outputs back to the inputs and updates all four projections
        /// </summary>
        public Quad[][] Backpropagate(Quad[][] outputGradients)
        {
            if (outputGradients == null) throw new ArgumentNullException(nameof(outputGradients));
            if (this.lastAttentionWeights == null) throw new InvalidOperationException("Propagate must be called before Backpropagate");

            int length = this.lastQueries.Length;
            Quad[][] concatenatedGradients = this.outputProjection.Backpropagate(outputGradients);

            Quad[][] queryGradients = TransformerUtils.CreateSequence(length, this.ModelDimension);
            Quad[][] keyGradients = TransformerUtils.CreateSequence(length, this.ModelDimension);
            Quad[][] valueGradients = TransformerUtils.CreateSequence(length, this.ModelDimension);

            for (int head = 0; head < this.headsCount; head++)
            {
                int offset = head * this.headDimension;
                Quad[][] attentionWeights = this.lastAttentionWeights[head];

                for (int i = 0; i < length; i++)
                {
                    Quad[] weights = attentionWeights[i];
                    Quad[] weightGradients = new Quad[length];
                    Quad weightedGradientSum = 0;

                    for (int j = 0; j < length; j++)
                    {
                        if (this.IsMasked(i, j))
                        {
                            continue;
                        }

                        Quad dot = 0;
                        for (int c = 0; c < this.headDimension; c++)
                        {
                            Quad gradient = concatenatedGradients[i][offset + c];
                            dot += gradient * this.lastValues[j][offset + c];
                            valueGradients[j][offset + c] += weights[j] * gradient;
                        }

                        weightGradients[j] = dot;
                        weightedGradientSum += dot * weights[j];
                    }

                    for (int j = 0; j < length; j++)
                    {
                        if (this.IsMasked(i, j))
                        {
                            continue;
                        }

                        // soft-max derivative followed by the 1 / sqrt(headDimension) scaling
                        Quad scoreGradient = weights[j] * (weightGradients[j] - weightedGradientSum) * this.scale;
                        for (int c = 0; c < this.headDimension; c++)
                        {
                            queryGradients[i][offset + c] += scoreGradient * this.lastKeys[j][offset + c];
                            keyGradients[j][offset + c] += scoreGradient * this.lastQueries[i][offset + c];
                        }
                    }
                }
            }

            Quad[][] inputGradients = this.queryProjection.Backpropagate(queryGradients);
            inputGradients = TransformerUtils.SequenceSum(inputGradients, this.keyProjection.Backpropagate(keyGradients));
            inputGradients = TransformerUtils.SequenceSum(inputGradients, this.valueProjection.Backpropagate(valueGradients));

            return inputGradients;
        }

        private bool IsMasked(int queryPosition, int keyPosition)
        {
            return this.useCausalMask && keyPosition > queryPosition;
        }
    }
}
