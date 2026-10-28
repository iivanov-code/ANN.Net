using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;

namespace ANN.Net.Transformers
{
    /// <summary>
    /// Dense projection y = x * W + b applied to every vector of a sequence.
    /// It is the vectorised counterpart of a fully connected layer of <see cref="Connections.ActiveSynapse"/>:
    /// every weight owns its optimizer and is updated as soon as its gradient is known
    /// </summary>
    [Serializable]
    internal class LinearProjection
    {
        private readonly Quad[,] weights;
        private readonly Quad[] biases;
        private readonly IOptimizer[,] weightOptimizers;
        private readonly IOptimizer[] biasOptimizers;
        private Quad[][] lastInputs;

        public LinearProjection(int inputsCount,
                                int outputsCount,
                                OptimizerTypes optimizerType,
                                WeightInitType initializationType,
                                ICollection<LearningRateOptimizer> optimizers)
        {
            if (inputsCount < 1) throw new ArgumentOutOfRangeException(nameof(inputsCount), "Must be greater than 0");
            if (outputsCount < 1) throw new ArgumentOutOfRangeException(nameof(outputsCount), "Must be greater than 0");
            if (optimizers == null) throw new ArgumentNullException(nameof(optimizers));

            this.weights = new Quad[inputsCount, outputsCount];
            this.weightOptimizers = new IOptimizer[inputsCount, outputsCount];
            this.biases = new Quad[outputsCount];
            this.biasOptimizers = new IOptimizer[outputsCount];

            for (int i = 0; i < inputsCount; i++)
            {
                for (int j = 0; j < outputsCount; j++)
                {
                    this.weights[i, j] = TransformerUtils.InitialWeight(initializationType, inputsCount, outputsCount);
                    this.weightOptimizers[i, j] = TransformerUtils.CreateOptimizer(optimizerType, optimizers);
                }
            }

            for (int j = 0; j < outputsCount; j++)
            {
                this.biases[j] = 0;
                this.biasOptimizers[j] = TransformerUtils.CreateOptimizer(optimizerType, optimizers);
            }
        }

        public int InputsCount => this.weights.GetLength(0);

        public int OutputsCount => this.weights.GetLength(1);

        public Quad[,] Weights => this.weights;

        public Quad[] Biases => this.biases;

        public Quad[][] Propagate(Quad[][] inputs)
        {
            if (inputs == null) throw new ArgumentNullException(nameof(inputs));

            int inputsCount = this.InputsCount;
            int outputsCount = this.OutputsCount;

            Quad[][] outputs = new Quad[inputs.Length][];
            for (int t = 0; t < inputs.Length; t++)
            {
                Quad[] input = inputs[t];
                if (input.Length != inputsCount)
                    throw new ArgumentException($"Expected vectors of size {inputsCount} but found {input.Length}", nameof(inputs));

                Quad[] output = new Quad[outputsCount];
                for (int j = 0; j < outputsCount; j++)
                {
                    Quad sum = this.biases[j];
                    for (int i = 0; i < inputsCount; i++)
                    {
                        sum += input[i] * this.weights[i, j];
                    }

                    output[j] = sum;
                }

                outputs[t] = output;
            }

            this.lastInputs = inputs;
            return outputs;
        }

        /// <summary>
        /// Propagates the gradient of the outputs back to the inputs and updates the weights
        /// </summary>
        /// <returns>Gradient with respect to the inputs of the last propagation</returns>
        public Quad[][] Backpropagate(Quad[][] outputGradients)
        {
            if (outputGradients == null) throw new ArgumentNullException(nameof(outputGradients));
            if (this.lastInputs == null) throw new InvalidOperationException("Propagate must be called before Backpropagate");
            if (outputGradients.Length != this.lastInputs.Length)
                throw new ArgumentException("Gradient sequence length differs from the propagated sequence length", nameof(outputGradients));

            int inputsCount = this.InputsCount;
            int outputsCount = this.OutputsCount;

            Quad[,] weightGradients = new Quad[inputsCount, outputsCount];
            Quad[] biasGradients = new Quad[outputsCount];
            Quad[][] inputGradients = TransformerUtils.CreateSequence(this.lastInputs.Length, inputsCount);

            for (int t = 0; t < outputGradients.Length; t++)
            {
                Quad[] outputGradient = outputGradients[t];
                if (outputGradient.Length != outputsCount)
                    throw new ArgumentException($"Expected gradients of size {outputsCount} but found {outputGradient.Length}", nameof(outputGradients));

                Quad[] input = this.lastInputs[t];
                Quad[] inputGradient = inputGradients[t];

                for (int j = 0; j < outputsCount; j++)
                {
                    Quad gradient = outputGradient[j];
                    biasGradients[j] += gradient;

                    for (int i = 0; i < inputsCount; i++)
                    {
                        inputGradient[i] += gradient * this.weights[i, j];
                        weightGradients[i, j] += input[i] * gradient;
                    }
                }
            }

            for (int i = 0; i < inputsCount; i++)
            {
                for (int j = 0; j < outputsCount; j++)
                {
                    this.weights[i, j] = this.weightOptimizers[i, j].Calculate(weightGradients[i, j], this.weights[i, j]);
                }
            }

            for (int j = 0; j < outputsCount; j++)
            {
                this.biases[j] = this.biasOptimizers[j].Calculate(biasGradients[j], this.biases[j]);
            }

            return inputGradients;
        }
    }
}
