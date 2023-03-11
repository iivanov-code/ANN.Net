using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
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

        public Quad[] Propagate(Quad[] values)
        {
            if (shouldNormalize)
            {
                Quad[] normalizedValues = new Quad[values.Length];
                for (int i = 0; i < values.Length; i++)
                {
                    normalizedValues[i] = NetworkUtils.Normalize(values[i], MinMaxValues[i].Item1, MinMaxValues[i].Item2);
                }
                values = normalizedValues;
            }

            for (int i = 0; i < values.Length; i++)
            {
                InputNeurons[i].Propagate(new NeuronPropagateEventArgs(values[i]));
            }

            if (InputNeurons.Count > values.Length && InputNeurons[values.Length] is BiasNeuron)
            {
                InputNeurons[values.Length].Propagate(new NeuronPropagateEventArgs(Quad.MaxValue));
            }

            return Values;
        }

        public Quad Backpropagate(Quad[] targets)
        {
            if (shouldNormalize)
            {
                Quad[] normalizedValues = new Quad[targets.Length];
                for (int i = 0; i < targets.Length; i++)
                {
                    normalizedValues[i] = NetworkUtils.Normalize(targets[i], MinMaxValues[i].Item1, MinMaxValues[i].Item2);
                }
                targets = normalizedValues;
            }

            Quad totalError = 0;
            Quad[] values = OutputNeurons.Select(x => x.Value).ToArray();
            for (int i = 0; i < targets.Length; i++)
            {
                Quad error = this.lossFunction.CalculateLoss(targets[i], values[i]);
                this.Errors[i] = error;
                OutputNeurons[i].Backpropagate(new BackpropagateEventArgs(error));
                totalError += error;
            }

            return totalError;
        }
    }
}
