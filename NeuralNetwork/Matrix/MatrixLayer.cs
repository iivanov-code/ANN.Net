using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Connections;
using ANN.Net.Neurons;
using ANN.Net.Optimizers;
using ANN.Net.Utils;

namespace ANN.Net.Matrix
{
    /// <summary>
    /// Weights from every input, and from the bias neuron of the previous layer, to every neuron of a layer
    /// </summary>
    [Serializable]
    internal sealed class MatrixLayer
    {
        private readonly IActivationFunction function;

        internal MatrixLayer(int inputs, bool bias, int outputs, ActivationTypes activation, bool activate,
                             OptimizerTypes weightOptimizer, OptimizerTypes biasOptimizer, ICollection<LearningRateOptimizer> optimizers)
        {
            Inputs = inputs;
            Outputs = outputs;
            Activation = activation;
            Activate = activate;
            function = NetworkUtils.GetActivation(activation);
            Weights = new Quad[checked(inputs * outputs)];
            Bias = bias ? new Quad[outputs] : null;
            WeightOptimizer = ArrayOptimizer.Create(weightOptimizer, Weights.Length, optimizers);
            BiasOptimizer = bias ? ArrayOptimizer.Create(biasOptimizer, outputs, optimizers) : null;
        }

        /// <summary>
        /// Training replica: the same weights, its own values of the last propagation, and optimizers that only collect
        /// gradients
        /// </summary>
        private MatrixLayer(MatrixLayer master, ReplicaContext context)
        {
            Inputs = master.Inputs;
            Outputs = master.Outputs;
            Activation = master.Activation;
            Activate = master.Activate;
            function = NetworkUtils.GetActivation(master.Activation);
            Weights = master.Weights;
            Bias = master.Bias;
            WeightOptimizer = context.Link(master.WeightOptimizer, master.Weights);
            BiasOptimizer = master.Bias == null ? null : context.Link(master.BiasOptimizer, master.Bias);
        }

        public MatrixLayer Replicate(ReplicaContext context) => new MatrixLayer(this, context);

        public int Inputs { get; }

        public int Outputs { get; }

        public ActivationTypes Activation { get; }

        /// <summary>
        /// False for the outputs of a soft-max network, which are not activated
        /// </summary>
        public bool Activate { get; }

        /// <summary>
        /// The weight from input i to neuron j is at i * <see cref="Outputs"/> + j
        /// </summary>
        public Quad[] Weights { get; }

        /// <summary>
        /// Weights from the bias neuron of the previous layer, or null when it has none
        /// </summary>
        public Quad[] Bias { get; }

        public ArrayOptimizer WeightOptimizer { get; }

        public ArrayOptimizer BiasOptimizer { get; }

        public Quad[] PreActivations { get; private set; }

        public Quad[] Values { get; private set; }

        public static MatrixLayer Create(int inputs, bool bias, bool activate, CommonLayerSettings settings, ICollection<LearningRateOptimizer> optimizers)
        {
            var layer = new MatrixLayer(inputs, bias, settings.NeuronsCount, settings.ActivationType, activate, settings.OptimizerType, settings.OptimizerType, optimizers);
            int fanIn = inputs + (bias ? 1 : 0);
            int outputs = settings.NeuronsCount;

            // the neuron-graph builder draws neuron by neuron, every neuron's inputs in order with the bias last
            for (int j = 0; j < outputs; j++)
            {
                for (int i = 0; i < fanIn; i++)
                {
                    Quad weight = settings.InitializationType switch
                    {
                        WeightInitType.Random => NetworkUtils.RandomWeight(),
                        WeightInitType.Xavier => NetworkUtils.GlorotWeight(fanIn, settings.NeuronsCount),
                        _ => throw new ArgumentException($"Unknown initialization {settings.InitializationType}", nameof(settings))
                    };

                    if (i < inputs) layer.Weights[(i * outputs) + j] = weight;
                    else layer.Bias[j] = weight;
                }
            }

            return layer;
        }

        public static MatrixLayer From(List<INeuron> previous, INeuron previousBias, List<INeuron> neurons, bool output, ICollection<LearningRateOptimizer> optimizers)
        {
            Type neuronType = output ? typeof(OutputNeuron) : typeof(HiddenNeuron);
            if (neurons.Count == 0 || neurons.Any(x => x.GetType() != neuronType))
                throw new NotSupportedException("Matrix networks support plain hidden and output neurons only");

            ActivationTypes activation = ((Neuron)neurons[0]).GetActivationType();
            bool activate = !output || ((OutputNeuron)neurons[0]).ApplyActivation;
            if (neurons.Any(x => ((Neuron)x).GetActivationType() != activation))
                throw new NotSupportedException("Matrix networks need the same activation for every neuron of a layer");

            List<BaseSynapse> synapses = neurons.SelectMany(x => x.Inputs).Select(x => x as BaseSynapse).ToList();
            if (synapses.Any(x => x?.GetType() != typeof(ActiveSynapse)))
                throw new NotSupportedException("Matrix networks support synapses with their own weight only");

            List<BaseSynapse> weights = synapses.Where(x => !ReferenceEquals(x.Input, previousBias)).ToList();
            List<BaseSynapse> biases = synapses.Where(x => ReferenceEquals(x.Input, previousBias)).ToList();
            (OptimizerTypes weightType, float weightRate) = CommonOptimizer(weights);
            (OptimizerTypes biasType, float biasRate) = biases.Count > 0 ? CommonOptimizer(biases) : (weightType, weightRate);

            var layer = new MatrixLayer(previous.Count, previousBias != null, neurons.Count, activation, activate, weightType, biasType, optimizers);
            layer.WeightOptimizer.LearningRate = weightRate;
            if (layer.BiasOptimizer != null) layer.BiasOptimizer.LearningRate = biasRate;

            var index = new Dictionary<INeuron, int>(ReferenceEqualityComparer.Instance);
            for (int i = 0; i < previous.Count; i++) index.Add(previous[i], i);

            var filled = new bool[layer.Weights.Length + (layer.Bias?.Length ?? 0)];
            for (int j = 0; j < neurons.Count; j++)
            {
                int expected = 0;
                foreach (BaseSynapse synapse in neurons[j].Inputs.Cast<BaseSynapse>())
                {
                    int position;
                    if (ReferenceEquals(synapse.Input, previousBias)) position = layer.Weights.Length + j;
                    else if (index.TryGetValue(synapse.Input, out int i) && i == expected++) position = (i * neurons.Count) + j;
                    else throw new NotSupportedException("Matrix networks need every neuron connected to all neurons of the previous layer, in order");

                    if (filled[position]) throw new NotSupportedException("Matrix networks support one synapse per pair of neurons");
                    filled[position] = true;

                    if (position < layer.Weights.Length) layer.Weights[position] = synapse.Weight;
                    else layer.Bias[j] = synapse.Weight;
                }
            }

            if (Array.IndexOf(filled, false) >= 0)
                throw new NotSupportedException("Matrix networks need fully connected layers");

            return layer;
        }

        private static (OptimizerTypes Type, float LearningRate) CommonOptimizer(List<BaseSynapse> synapses)
        {
            OptimizerTypes type = synapses[0].Optimizer.GetOptimizerType();
            float rate = RateOf(synapses[0].Optimizer);
            if (synapses.Any(x => x.Optimizer.GetOptimizerType() != type || !RateOf(x.Optimizer).Equals(rate)))
                throw new NotSupportedException("Matrix networks need one optimizer type and learning rate for the weights and one for the biases of a layer");

            return (type, rate);
        }

        private static float RateOf(IOptimizer optimizer) => optimizer is LearningRateOptimizer withRate ? withRate.LearningRate : 0;

        public Quad[] Propagate(Quad[] input)
        {
            var pre = new Quad[Outputs];
            var values = new Quad[Outputs];
            for (int j = 0; j < Outputs; j++)
            {
                Quad sum = 0;
                for (int i = 0; i < Inputs; i++)
                {
                    sum += input[i] * Weights[(i * Outputs) + j];
                }

                if (Bias != null) sum += 1 * Bias[j];

                pre[j] = sum;
                Quad value = sum;
                values[j] = Activate ? function.Activation(ref value) : sum;
            }

            PreActivations = pre;
            Values = values;
            return values;
        }

        /// <summary>
        /// Takes the gradient of the loss with respect to the layer's values, updates the weights and returns the
        /// gradient with respect to the inputs, computed with the weights before the update
        /// </summary>
        public Quad[] Backpropagate(Quad[] valueGradient, Quad[] input)
        {
            var delta = new Quad[Outputs];
            for (int j = 0; j < Outputs; j++)
            {
                delta[j] = Activate ? valueGradient[j] * function.Derivative(PreActivations[j]) : valueGradient[j];
            }

            var inputGradient = new Quad[Inputs];
            for (int i = 0; i < Inputs; i++)
            {
                Quad sum = 0;
                int row = i * Outputs;
                for (int j = 0; j < Outputs; j++)
                {
                    sum += delta[j] * Weights[row + j];
                }

                inputGradient[i] = sum;

                for (int j = 0; j < Outputs; j++)
                {
                    Weights[row + j] = WeightOptimizer.Calculate(row + j, delta[j] * input[i], Weights[row + j]);
                }
            }

            WeightOptimizer.Advance();

            if (Bias != null)
            {
                for (int j = 0; j < Outputs; j++)
                {
                    Bias[j] = BiasOptimizer.Calculate(j, delta[j] * 1, Bias[j]);
                }

                BiasOptimizer.Advance();
            }

            return inputGradient;
        }
    }
}
