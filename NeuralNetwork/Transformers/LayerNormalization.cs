using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;

namespace ANN.Net.Transformers
{
    /// <summary>
    /// Normalizes every vector of a sequence to zero mean and unit variance,
    /// then rescales it with a trainable gain and offset per dimension
    /// </summary>
    [Serializable]
    internal class LayerNormalization
    {
        private const float EPSILON = 1e-5f;

        private readonly Quad[] gains;
        private readonly Quad[] offsets;
        private readonly IOptimizer[] gainOptimizers;
        private readonly IOptimizer[] offsetOptimizers;
        private Quad[][] lastNormalized;
        private Quad[] lastInverseDeviations;

        public LayerNormalization(int dimension, OptimizerTypes optimizerType, ICollection<LearningRateOptimizer> optimizers)
        {
            if (dimension < 1) throw new ArgumentOutOfRangeException(nameof(dimension), "Must be greater than 0");
            if (optimizers == null) throw new ArgumentNullException(nameof(optimizers));

            this.gains = new Quad[dimension];
            this.offsets = new Quad[dimension];
            this.gainOptimizers = new IOptimizer[dimension];
            this.offsetOptimizers = new IOptimizer[dimension];

            for (int i = 0; i < dimension; i++)
            {
                this.gains[i] = 1;
                this.offsets[i] = 0;
                this.gainOptimizers[i] = TransformerUtils.CreateOptimizer(optimizerType, optimizers);
                this.offsetOptimizers[i] = TransformerUtils.CreateOptimizer(optimizerType, optimizers);
            }
        }

        public int Dimension => this.gains.Length;

        public Quad[] Gains => this.gains;

        public Quad[] Offsets => this.offsets;

        public Quad[][] Propagate(Quad[][] inputs)
        {
            if (inputs == null) throw new ArgumentNullException(nameof(inputs));

            int dimension = this.Dimension;
            Quad[][] outputs = new Quad[inputs.Length][];
            Quad[][] normalized = new Quad[inputs.Length][];
            Quad[] inverseDeviations = new Quad[inputs.Length];

            for (int t = 0; t < inputs.Length; t++)
            {
                Quad[] input = inputs[t];
                if (input.Length != dimension)
                    throw new ArgumentException($"Expected vectors of size {dimension} but found {input.Length}", nameof(inputs));

                Quad mean = 0;
                for (int i = 0; i < dimension; i++)
                {
                    mean += input[i];
                }
                mean /= dimension;

                Quad variance = 0;
                for (int i = 0; i < dimension; i++)
                {
                    Quad centered = input[i] - mean;
                    variance += centered * centered;
                }
                variance /= dimension;

                Quad inverseDeviation = (Quad)(1 / Math.Sqrt(variance + EPSILON));
                inverseDeviations[t] = inverseDeviation;

                normalized[t] = new Quad[dimension];
                outputs[t] = new Quad[dimension];
                for (int i = 0; i < dimension; i++)
                {
                    normalized[t][i] = (input[i] - mean) * inverseDeviation;
                    outputs[t][i] = normalized[t][i] * this.gains[i] + this.offsets[i];
                }
            }

            this.lastNormalized = normalized;
            this.lastInverseDeviations = inverseDeviations;
            return outputs;
        }

        /// <summary>
        /// Propagates the gradient of the outputs back to the inputs and updates gains and offsets
        /// </summary>
        public Quad[][] Backpropagate(Quad[][] outputGradients)
        {
            if (outputGradients == null) throw new ArgumentNullException(nameof(outputGradients));
            if (this.lastNormalized == null) throw new InvalidOperationException("Propagate must be called before Backpropagate");
            if (outputGradients.Length != this.lastNormalized.Length)
                throw new ArgumentException("Gradient sequence length differs from the propagated sequence length", nameof(outputGradients));

            int dimension = this.Dimension;
            Quad[] gainGradients = new Quad[dimension];
            Quad[] offsetGradients = new Quad[dimension];
            Quad[][] inputGradients = new Quad[outputGradients.Length][];

            for (int t = 0; t < outputGradients.Length; t++)
            {
                Quad[] outputGradient = outputGradients[t];
                if (outputGradient.Length != dimension)
                    throw new ArgumentException($"Expected gradients of size {dimension} but found {outputGradient.Length}", nameof(outputGradients));

                Quad[] normalized = this.lastNormalized[t];
                Quad[] normalizedGradient = new Quad[dimension];
                Quad gradientMean = 0;
                Quad weightedGradientMean = 0;

                for (int i = 0; i < dimension; i++)
                {
                    gainGradients[i] += outputGradient[i] * normalized[i];
                    offsetGradients[i] += outputGradient[i];

                    normalizedGradient[i] = outputGradient[i] * this.gains[i];
                    gradientMean += normalizedGradient[i];
                    weightedGradientMean += normalizedGradient[i] * normalized[i];
                }
                gradientMean /= dimension;
                weightedGradientMean /= dimension;

                inputGradients[t] = new Quad[dimension];
                for (int i = 0; i < dimension; i++)
                {
                    inputGradients[t][i] = this.lastInverseDeviations[t] * (normalizedGradient[i] - gradientMean - normalized[i] * weightedGradientMean);
                }
            }

            for (int i = 0; i < dimension; i++)
            {
                this.gains[i] = this.gainOptimizers[i].Calculate(gainGradients[i], this.gains[i]);
                this.offsets[i] = this.offsetOptimizers[i].Calculate(offsetGradients[i], this.offsets[i]);
            }

            return inputGradients;
        }
    }
}
