using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.IdentityGenerators;

namespace ANN.Net.Transformers
{
    /// <summary>
    /// Position-wise feed forward sub-layer built from neuron objects: an identity input layer,
    /// one activated hidden layer and a linear output layer, fully connected with bias neurons.
    /// The same <see cref="Network"/> is applied to every vector of the sequence, one position at a time
    /// </summary>
    [Serializable]
    internal sealed class NeuronFeedForwardNetwork : IFeedForwardNetwork
    {
        private readonly Network network;
        private readonly int modelDimension;
        private Quad[][] lastInputs;

        public NeuronFeedForwardNetwork(int modelDimension,
                                        int hiddenDimension,
                                        ActivationTypes activationType,
                                        OptimizerTypes optimizerType,
                                        WeightInitType initializationType,
                                        ICollection<LearningRateOptimizer> optimizers)
        {
            if (modelDimension < 1) throw new ArgumentOutOfRangeException(nameof(modelDimension), "Must be greater than 0");
            if (hiddenDimension < 1) throw new ArgumentOutOfRangeException(nameof(hiddenDimension), "Must be greater than 0");
            if (optimizers == null) throw new ArgumentNullException(nameof(optimizers));

            this.modelDimension = modelDimension;

            NetworkSettings settings = new NetworkSettings
            {
                IdentityGenerator = new IncrementalGenerator(),
                LossType = LossFunctionTypes.Squared,
                OutputApplySoftMax = false,
                ShouldNormalize = false,
                InputSettings = new InputLayerSettings
                {
                    NeuronsCount = (ushort)modelDimension,
                    ActivationType = ActivationTypes.Identity,
                    HasBiasNeuron = true,
                    InitializationType = initializationType,
                    OptimizerType = optimizerType
                },
                OutputSettings = new OutputLayerSettings
                {
                    NeuronsCount = (ushort)modelDimension,
                    ActivationType = ActivationTypes.Identity,
                    HasBiasNeuron = false,
                    InitializationType = initializationType,
                    OptimizerType = optimizerType
                }
            };

            RecurrentLayerSettings hidden = new RecurrentLayerSettings
            {
                NeuronsCount = (ushort)hiddenDimension,
                LayersCount = 1,
                CellType = NeuronType.Normal,
                PrevLayerConnectionType = LayerConnectionType.FullyConnected,
                ActivationType = activationType,
                HasBiasNeuron = true,
                InitializationType = initializationType,
                OptimizerType = optimizerType
            };

            this.network = (Network)NetworkFactory.BuildFFN(settings, hidden);

            foreach (LearningRateOptimizer optimizer in this.network.Optimizers)
            {
                optimizers.Add(optimizer);
            }
        }

        public Network Network => this.network;

        public Quad[][] Propagate(Quad[][] inputs)
        {
            if (inputs == null) throw new ArgumentNullException(nameof(inputs));

            Quad[][] outputs = new Quad[inputs.Length][];
            for (int t = 0; t < inputs.Length; t++)
            {
                if (inputs[t].Length != this.modelDimension)
                    throw new ArgumentException($"Expected vectors of size {this.modelDimension} but found {inputs[t].Length}", nameof(inputs));

                outputs[t] = this.network.Propagate(inputs[t]);
            }

            this.lastInputs = inputs;
            return outputs;
        }

        /// <summary>
        /// Every position is propagated again to restore the neuron state of that position and then
        /// backpropagated with targets chosen so that value - target equals the upstream gradient.
        /// Weights therefore update after every position, in the order of the sequence
        /// </summary>
        public Quad[][] Backpropagate(Quad[][] outputGradients)
        {
            if (outputGradients == null) throw new ArgumentNullException(nameof(outputGradients));
            if (this.lastInputs == null) throw new InvalidOperationException("Propagate must be called before Backpropagate");
            if (outputGradients.Length != this.lastInputs.Length)
                throw new ArgumentException("Gradient sequence length differs from the propagated sequence length", nameof(outputGradients));

            Quad[][] inputGradients = new Quad[outputGradients.Length][];
            for (int t = 0; t < outputGradients.Length; t++)
            {
                Quad[] gradient = outputGradients[t];
                if (gradient.Length != this.modelDimension)
                    throw new ArgumentException($"Expected gradients of size {this.modelDimension} but found {gradient.Length}", nameof(outputGradients));

                Quad[] values = this.network.Propagate(this.lastInputs[t]);
                Quad[] targets = new Quad[values.Length];
                for (int i = 0; i < values.Length; i++)
                {
                    targets[i] = values[i] - gradient[i];
                }

                this.network.Backpropagate(targets);
                inputGradients[t] = this.network.InputGradients;
            }

            return inputGradients;
        }
    }
}
