using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Utils;

namespace ANN.Net.Transformers
{
    /// <summary>
    /// Position-wise feed forward network: expand, activate, contract.
    /// The same weights are applied to every vector of the sequence
    /// </summary>
    [Serializable]
    internal class FeedForwardNetwork : IFeedForwardNetwork
    {
        private readonly LinearProjection expansion;
        private readonly LinearProjection contraction;
        private readonly IActivationFunction activation;
        private Quad[][] lastHidden;

        public FeedForwardNetwork(int modelDimension,
                                  int hiddenDimension,
                                  ActivationTypes activationType,
                                  OptimizerTypes optimizerType,
                                  WeightInitType initializationType,
                                  ICollection<LearningRateOptimizer> optimizers)
        {
            this.expansion = new LinearProjection(modelDimension, hiddenDimension, optimizerType, initializationType, optimizers);
            this.contraction = new LinearProjection(hiddenDimension, modelDimension, optimizerType, initializationType, optimizers);
            this.activation = NetworkUtils.GetActivation(activationType);
        }

        public Quad[][] Propagate(Quad[][] inputs)
        {
            Quad[][] hidden = this.expansion.Propagate(inputs);
            Quad[][] activated = new Quad[hidden.Length][];

            for (int t = 0; t < hidden.Length; t++)
            {
                activated[t] = new Quad[hidden[t].Length];
                for (int i = 0; i < hidden[t].Length; i++)
                {
                    Quad value = hidden[t][i];
                    activated[t][i] = this.activation.Activation(ref value);
                }
            }

            this.lastHidden = hidden;
            return this.contraction.Propagate(activated);
        }

        /// <summary>
        /// Propagates the gradient of the outputs back to the inputs and updates both projections
        /// </summary>
        public Quad[][] Backpropagate(Quad[][] outputGradients)
        {
            if (this.lastHidden == null) throw new InvalidOperationException("Propagate must be called before Backpropagate");

            Quad[][] hiddenGradients = this.contraction.Backpropagate(outputGradients);

            for (int t = 0; t < hiddenGradients.Length; t++)
            {
                for (int i = 0; i < hiddenGradients[t].Length; i++)
                {
                    // the derivative is evaluated on the value before activation, as Neuron.FeedbackError does
                    hiddenGradients[t][i] *= this.activation.Derivative(this.lastHidden[t][i]);
                }
            }

            return this.expansion.Backpropagate(hiddenGradients);
        }
    }
}
