using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Transformers;
using ANN.Net.Optimizers;

namespace ANN.Net.Convolutional
{
    /// <summary>
    /// Fully connected layer over the flattened input
    /// </summary>
    [Serializable]
    internal sealed class DenseLayer : IConvolutionalLayer
    {
        private readonly Activation activation;
        private Quad[] lastInput;

        public DenseLayer(Shape inputShape, int neuronsCount, ActivationTypes activationType, OptimizerTypes optimizerType,
                          WeightInitType initializationType, ICollection<LearningRateOptimizer> optimizers)
        {
            if (neuronsCount < 1) throw new ArgumentOutOfRangeException(nameof(neuronsCount), neuronsCount, "Must be at least 1");

            InputShape = inputShape;
            OutputShape = new Shape(neuronsCount, 1, 1);
            activation = new Activation(activationType);

            int inputs = inputShape.Size;
            Weights = new Parameters(inputs * neuronsCount, optimizerType, optimizers, () => TransformerUtils.InitialWeight(initializationType, inputs, neuronsCount));
            Biases = new Parameters(neuronsCount, optimizerType, optimizers, () => 0);
        }

        private DenseLayer(DenseLayer master, ReplicaContext context)
        {
            InputShape = master.InputShape;
            OutputShape = master.OutputShape;
            activation = new Activation(master.ActivationType);
            Weights = new Parameters(master.Weights, context);
            Biases = new Parameters(master.Biases, context);
        }

        public IConvolutionalLayer Replicate(ReplicaContext context) => new DenseLayer(this, context);

        public Shape InputShape { get; }

        public Shape OutputShape { get; }

        public ActivationTypes ActivationType => activation.Type;

        /// <summary>
        /// Weights in input, output order: the weight from input i to neuron j is at i * neurons + j
        /// </summary>
        public Parameters Weights { get; }

        public Parameters Biases { get; }

        public Quad[] Propagate(Quad[] input)
        {
            lastInput = input;
            int inputs = InputShape.Size, outputs = OutputShape.Size;
            var sums = new Quad[outputs];

            for (int j = 0; j < outputs; j++)
            {
                Quad sum = Biases.Values[j];
                for (int i = 0; i < inputs; i++)
                {
                    sum += input[i] * Weights.Values[(i * outputs) + j];
                }

                sums[j] = sum;
            }

            return activation.Apply(sums);
        }

        public Quad[] Backpropagate(Quad[] outputGradient)
        {
            if (lastInput == null) throw new InvalidOperationException("Propagate must be called before Backpropagate");

            Quad[] gradient = activation.Backpropagate(outputGradient);
            int inputs = InputShape.Size, outputs = OutputShape.Size;
            var inputGradient = new Quad[inputs];
            var weightGradients = new Quad[Weights.Values.Length];

            for (int i = 0; i < inputs; i++)
            {
                for (int j = 0; j < outputs; j++)
                {
                    int index = (i * outputs) + j;
                    weightGradients[index] = gradient[j] * lastInput[i];
                    inputGradient[i] += gradient[j] * Weights.Values[index];
                }
            }

            Weights.Update(weightGradients);
            Biases.Update(gradient);
            return inputGradient;
        }
    }
}
