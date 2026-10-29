using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Abstractions.Models;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Neurons;
using ANN.Net.Utils;

namespace ANN.Net
{
    [Serializable]
    internal sealed class Network : ANNetwork, INetwork
    {
        public IReadOnlyList<Tuple<Quad, Quad>> MinMaxValues { get; }

        private readonly bool shouldNormalize;
        private readonly bool applySoftMax;
        public ILayer<IInputNeuron> InputNeurons;
        public ICollection<ILayer<IHiddenNeuron>> HiddenLayers;
        public ILayer<IOutputNeuron> OutputNeurons;
        public ICollection<LearningRateOptimizer> Optimizers { get; }

        private ILossFunction lossFunction;

        public Network(NetworkSettings settings)
        {
            this.MinMaxValues = settings.MinMaxValues;
            this.shouldNormalize = settings.ShouldNormalize;
            this.Errors = new Quad[settings.OutputSettings.NeuronsCount];
            this.applySoftMax = settings.OutputApplySoftMax;
            this.Optimizers = new List<LearningRateOptimizer>();
            this.lossFunction = NetworkUtils.GetLossFunction(settings.LossType);
        }

        public Quad[] Values
        {
            get
            {
                if (applySoftMax)
                {
                    Quad sum = OutputNeurons.Sum(x => x.Value);
                    return OutputNeurons.Select(x => x.Value / sum).ToArray();
                }
                else
                {
                    return OutputNeurons.Select(x => x.Value).ToArray();
                }
            }
        }

        public Quad[] Errors { get; }

        /// <summary>
        /// Gradient of the loss with respect to every input value, from the last backpropagation
        /// </summary>
        public Quad[] InputGradients => InputNeurons.OfType<InputNeuron>().Select(n => n.Gradient).ToArray();

        public Quad[] Propagate(Quad[] values)
        {
            if (values == null) throw new ArgumentNullException(nameof(values));

            int inputsCount = InputNeurons.Count(n => !(n is BiasNeuron));
            if (values.Length != inputsCount)
                throw new ArgumentException($"Expected {inputsCount} input values but found {values.Length}", nameof(values));

            if (shouldNormalize)
            {
                Quad[] normalizedValues = new Quad[values.Length];
                for (int i = 0; i < values.Length; i++)
                {
                    normalizedValues[i] = NetworkUtils.Normalize(values[i], MinMaxValues[i].Item1, MinMaxValues[i].Item2);
                }
                values = normalizedValues;
            }

            int index = 0;
            foreach (IInputNeuron neuron in InputNeurons)
            {
                if (neuron is BiasNeuron) continue;
                neuron.Propagate(new NeuronPropagateEventArgs(values[index++]));
            }

            // bias neurons have no inputs, so the network fires them itself on every propagation
            foreach (IInputNeuron neuron in InputNeurons)
            {
                if (neuron is BiasNeuron bias) bias.Propagate(new NeuronPropagateEventArgs(1));
            }

            if (HiddenLayers != null)
            {
                foreach (ILayer<IHiddenNeuron> layer in HiddenLayers)
                {
                    foreach (IHiddenNeuron neuron in layer)
                    {
                        if (neuron is BiasNeuron bias) bias.Propagate(new NeuronPropagateEventArgs(1));
                    }
                }
            }

            return Values;
        }

        public Quad Backpropagate(Quad[] targets)
        {
            if (targets == null) throw new ArgumentNullException(nameof(targets));
            if (targets.Length != OutputNeurons.Count)
                throw new ArgumentException($"Expected {OutputNeurons.Count} targets but found {targets.Length}", nameof(targets));

            if (shouldNormalize)
            {
                Quad[] normalizedValues = new Quad[targets.Length];
                for (int i = 0; i < targets.Length; i++)
                {
                    normalizedValues[i] = NetworkUtils.Normalize(targets[i], MinMaxValues[i].Item1, MinMaxValues[i].Item2);
                }
                targets = normalizedValues;
            }

            Quad[] values = OutputNeurons.Select(x => x.Value).ToArray();
            var results = new List<ErrorResult>(targets.Length);
            for (int i = 0; i < targets.Length; i++)
            {
                Quad error = values[i] - targets[i];
                this.Errors[i] = error;
                OutputNeurons[i].Backpropagate(new BackpropagateEventArgs(error));
                results.Add(new ErrorResult { Target = targets[i], Predicted = values[i] });
            }

            return this.lossFunction.CalculateLoss(results);
        }
    }
}
