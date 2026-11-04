using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Abstractions.Models;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Neurons;
using ANN.Net.Utils;

namespace ANN.Net.Matrix
{
    /// <summary>
    /// Fully connected layered network with one weight matrix and one bias vector per layer. It reproduces the
    /// neuron-graph <see cref="Network"/> operation by operation: the same normalization, input activation, sums in the
    /// same order (inputs first, the bias last), divide-by-sum soft-max, loss, gradients computed with the weights
    /// before their update, and the same optimizer rules. Built with the same settings and seed it starts from the same
    /// weights, so both networks give bit-identical results
    /// </summary>
    [Serializable]
    internal sealed class MatrixNetwork : ANNetwork, IMatrixNetwork
    {
        private readonly ILossFunction lossFunction;
        private readonly IActivationFunction inputFunction;
        private Quad[] inputPreActivations;
        private Quad[] inputValues;

        internal MatrixNetwork(int inputCount, bool inputBias, ActivationTypes inputActivation, bool shouldNormalize, IReadOnlyList<Tuple<Quad, Quad>> minMax,
                               bool softMax, LossFunctionTypes lossType)
        {
            InputCount = inputCount;
            InputBias = inputBias;
            InputActivation = inputActivation;
            inputFunction = NetworkUtils.GetActivation(inputActivation);
            ShouldNormalize = shouldNormalize;
            MinMaxValues = minMax;
            OutputApplySoftMax = softMax;
            LossType = lossType;
            lossFunction = NetworkUtils.GetLossFunction(lossType);
            Optimizers = new List<LearningRateOptimizer>();
            Layers = new List<MatrixLayer>();
        }

        internal int InputCount { get; }

        internal bool InputBias { get; }

        internal ActivationTypes InputActivation { get; }

        internal bool ShouldNormalize { get; }

        internal bool OutputApplySoftMax { get; }

        internal LossFunctionTypes LossType { get; }

        internal List<MatrixLayer> Layers { get; }

        public IReadOnlyList<Tuple<Quad, Quad>> MinMaxValues { get; }

        public ICollection<LearningRateOptimizer> Optimizers { get; }

        public long ParameterCount => Layers.Sum(x => (long)x.Weights.Length + (x.Bias?.Length ?? 0));

        public Quad[] Values { get; private set; } = Array.Empty<Quad>();

        public Quad[] Errors { get; private set; } = Array.Empty<Quad>();

        public Quad[] InputGradients { get; private set; } = Array.Empty<Quad>();

        /// <summary>
        /// Builds the network <c>NetworkFactory.BuildFFN</c> would build, drawing the initial weights in the same order
        /// </summary>
        public static MatrixNetwork Build(NetworkSettings settings, IEnumerable<RecurrentLayerSettings> hiddenSettings)
        {
            ArgumentNullException.ThrowIfNull(settings);
            using IDisposable seed = NetworkUtils.SeedRandom(settings.Seed);

            var network = new MatrixNetwork(settings.InputSettings.NeuronsCount, settings.InputSettings.HasBiasNeuron, settings.InputSettings.ActivationType,
                                            settings.ShouldNormalize, settings.MinMaxValues, settings.OutputApplySoftMax, settings.LossType);

            int previous = settings.InputSettings.NeuronsCount;
            bool previousBias = settings.InputSettings.HasBiasNeuron;

            foreach (RecurrentLayerSettings layer in hiddenSettings ?? Enumerable.Empty<RecurrentLayerSettings>())
            {
                if (layer.CellType != NeuronType.Normal || layer.PrevLayerConnectionType != LayerConnectionType.FullyConnected)
                    throw new NotSupportedException("Matrix networks support fully connected layers of normal neurons only");

                for (int i = 0; i < layer.LayersCount; i++)
                {
                    network.Layers.Add(MatrixLayer.Create(previous, previousBias, true, layer, network.Optimizers));
                    previous = layer.NeuronsCount;
                    previousBias = layer.HasBiasNeuron;
                }
            }

            network.Layers.Add(MatrixLayer.Create(previous, previousBias, !settings.OutputApplySoftMax, settings.OutputSettings, network.Optimizers));
            return network;
        }

        /// <summary>
        /// Copies the weights, optimizer types and learning rates of a fully connected neuron-graph network. The state
        /// of optimizers such as Adam starts afresh
        /// </summary>
        public static MatrixNetwork FromNetwork(Network source)
        {
            ArgumentNullException.ThrowIfNull(source);

            List<INeuron> inputs = source.InputNeurons.Where(x => x is not BiasNeuron).Cast<INeuron>().ToList();
            INeuron inputBias = source.InputNeurons.FirstOrDefault(x => x is BiasNeuron);
            if (inputs.Count == 0 || inputs.Any(x => x.GetType() != typeof(InputNeuron)))
                throw new NotSupportedException("Matrix networks support plain input neurons only");

            ActivationTypes inputActivation = ((Neuron)inputs[0]).GetActivationType();
            if (inputs.Any(x => ((Neuron)x).GetActivationType() != inputActivation))
                throw new NotSupportedException("Matrix networks need the same activation for every input");

            var network = new MatrixNetwork(inputs.Count, inputBias != null, inputActivation, source.ShouldNormalize, source.MinMaxValues,
                                            source.ApplySoftMax, source.LossType);

            List<List<INeuron>> layers = (source.HiddenLayers ?? Enumerable.Empty<ILayer<IHiddenNeuron>>())
                .Select(layer => layer.Cast<INeuron>().ToList())
                .Append(source.OutputNeurons.Cast<INeuron>().ToList())
                .ToList();

            List<INeuron> previous = inputs;
            INeuron previousBias = inputBias;
            for (int l = 0; l < layers.Count; l++)
            {
                List<INeuron> neurons = layers[l].Where(x => x is not BiasNeuron).ToList();
                network.Layers.Add(MatrixLayer.From(previous, previousBias, neurons, l == layers.Count - 1, network.Optimizers));

                previous = neurons;
                previousBias = layers[l].FirstOrDefault(x => x is BiasNeuron);
            }

            return network;
        }

        public Quad[] Propagate(params Quad[] values)
        {
            ArgumentNullException.ThrowIfNull(values);
            if (values.Length != InputCount)
                throw new ArgumentException($"Expected {InputCount} input values but found {values.Length}", nameof(values));

            var pre = new Quad[InputCount];
            var activated = new Quad[InputCount];
            for (int i = 0; i < InputCount; i++)
            {
                Quad value = ShouldNormalize ? NetworkUtils.Normalize(values[i], MinMaxValues[i].Item1, MinMaxValues[i].Item2) : values[i];
                pre[i] = value;
                activated[i] = inputFunction.Activation(ref value);
            }

            inputPreActivations = pre;
            inputValues = activated;

            Quad[] current = activated;
            foreach (MatrixLayer layer in Layers)
            {
                current = layer.Propagate(current);
            }

            Values = OutputValues(current);
            return Values;
        }

        /// <summary>
        /// The network's soft-max divides every output by the sum of all outputs, summed as the neuron-graph network does
        /// </summary>
        private Quad[] OutputValues(Quad[] outputs)
        {
            if (!OutputApplySoftMax) return (Quad[])outputs.Clone();

            float sum = outputs.Sum(x => (float)x);
            return outputs.Select(x => x / sum).ToArray();
        }

        public Quad Backpropagate(params Quad[] targets)
        {
            ArgumentNullException.ThrowIfNull(targets);
            if (inputValues == null) throw new InvalidOperationException("Propagate must be called before Backpropagate");

            MatrixLayer last = Layers[^1];
            if (targets.Length != last.Outputs)
                throw new ArgumentException($"Expected {last.Outputs} targets but found {targets.Length}", nameof(targets));

            if (ShouldNormalize)
            {
                targets = targets.Select((x, i) => NetworkUtils.Normalize(x, MinMaxValues[i].Item1, MinMaxValues[i].Item2)).ToArray();
            }

            var errors = new Quad[targets.Length];
            var results = new List<ErrorResult>(targets.Length);
            for (int i = 0; i < targets.Length; i++)
            {
                errors[i] = last.Values[i] - targets[i];
                results.Add(new ErrorResult { Target = targets[i], Predicted = last.Values[i] });
            }

            Errors = errors;
            BackpropagateLayers(errors);
            return lossFunction.CalculateLoss(results);
        }

        public int InputSize => InputCount;

        public int OutputSize => Layers[^1].Outputs;

        /// <summary>
        /// The gradient is taken with respect to the outputs <see cref="Propagate"/> returned, through the division by
        /// their sum of a soft-max network. The returned gradient is with respect to the inputs before normalization
        /// </summary>
        public Quad[] BackpropagateGradient(Quad[] outputGradients)
        {
            GradientUtils.CheckLength(outputGradients, OutputSize, nameof(outputGradients));
            if (inputValues == null) throw new InvalidOperationException("Propagate must be called before BackpropagateGradient");

            Quad[] gradient = OutputApplySoftMax ? GradientUtils.ThroughDivideBySum(outputGradients, Layers[^1].Values) : (Quad[])outputGradients.Clone();
            Errors = gradient;
            BackpropagateLayers(gradient);

            if (!ShouldNormalize) return (Quad[])InputGradients.Clone();
            return InputGradients.Select((g, i) => g / (MinMaxValues[i].Item2 - MinMaxValues[i].Item1)).ToArray();
        }

        private void BackpropagateLayers(Quad[] gradient)
        {
            for (int l = Layers.Count - 1; l >= 0; l--)
            {
                gradient = Layers[l].Backpropagate(gradient, l == 0 ? inputValues : Layers[l - 1].Values);
            }

            // the input neurons report the gradient before their activation
            InputGradients = gradient.Select((g, i) => g * inputFunction.Derivative(inputPreActivations[i])).ToArray();
        }
    }
}
