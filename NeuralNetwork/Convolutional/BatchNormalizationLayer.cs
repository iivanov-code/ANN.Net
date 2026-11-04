using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Optimizers;

namespace ANN.Net.Convolutional
{
    /// <summary>
    /// Batch normalization of every channel over its positions, followed by an activation. In training mode the mean and
    /// (biased) variance of the current image are used and the running averages are updated with the unbiased variance,
    /// as in PyTorch with a batch of one; otherwise the running averages are used
    /// </summary>
    [Serializable]
    internal sealed class BatchNormalizationLayer : IConvolutionalLayer
    {
        private readonly NetworkMode mode;
        private readonly Activation activation;
        private Quad[] normalized;
        private double[] inverseDeviations;
        private bool trainedLast;

        /// <summary>
        /// Replicas only: the mean and unbiased variance of every training propagation, which the network applies to
        /// the running averages after the batch, in sample order
        /// </summary>
        private readonly List<(double[] Mean, double[] Variance)> recorded;

        public BatchNormalizationLayer(Shape shape, ActivationTypes activationType, float momentum, float epsilon, NetworkMode mode,
                                       OptimizerTypes optimizerType, ICollection<LearningRateOptimizer> optimizers)
        {
            if (!(momentum >= 0 && momentum <= 1)) throw new ArgumentOutOfRangeException(nameof(momentum), momentum, "Must be in [0, 1]");
            if (!(epsilon > 0)) throw new ArgumentOutOfRangeException(nameof(epsilon), epsilon, "Must be positive");

            InputShape = shape;
            OutputShape = shape;
            Momentum = momentum;
            Epsilon = epsilon;
            this.mode = mode ?? throw new ArgumentNullException(nameof(mode));
            activation = new Activation(activationType);

            Gamma = new Parameters(shape.Channels, optimizerType, optimizers, () => 1);
            Beta = new Parameters(shape.Channels, optimizerType, optimizers, () => 0);
            RunningMean = new Quad[shape.Channels];
            RunningVariance = new Quad[shape.Channels];
            Array.Fill(RunningVariance, (Quad)1);
        }

        private BatchNormalizationLayer(BatchNormalizationLayer master, ReplicaContext context)
        {
            InputShape = master.InputShape;
            OutputShape = master.OutputShape;
            Momentum = master.Momentum;
            Epsilon = master.Epsilon;
            mode = master.mode;
            activation = new Activation(master.ActivationType);
            Gamma = new Parameters(master.Gamma, context);
            Beta = new Parameters(master.Beta, context);
            RunningMean = master.RunningMean;
            RunningVariance = master.RunningVariance;

            recorded = new List<(double[] Mean, double[] Variance)>();
            int index = context.LayerState.Count;
            context.LayerState.Add(recorded);
            context.Finishers.Add(contexts =>
            {
                foreach (ReplicaContext replica in contexts)
                {
                    var records = (List<(double[] Mean, double[] Variance)>)replica.LayerState[index];
                    foreach ((double[] mean, double[] variance) in records)
                    {
                        for (int c = 0; c < mean.Length; c++) master.UpdateRunning(c, mean[c], variance[c]);
                    }

                    records.Clear();
                }
            });
        }

        public IConvolutionalLayer Replicate(ReplicaContext context) => new BatchNormalizationLayer(this, context);

        private void UpdateRunning(int c, double mean, double unbiased)
        {
            RunningMean[c] = (Quad)(((1 - Momentum) * (float)RunningMean[c]) + (Momentum * mean));
            RunningVariance[c] = (Quad)(((1 - Momentum) * (float)RunningVariance[c]) + (Momentum * unbiased));
        }

        public Shape InputShape { get; }

        public Shape OutputShape { get; }

        public float Momentum { get; }

        public float Epsilon { get; }

        public ActivationTypes ActivationType => activation.Type;

        public Parameters Gamma { get; }

        public Parameters Beta { get; }

        public Quad[] RunningMean { get; }

        public Quad[] RunningVariance { get; }

        public Quad[] Propagate(Quad[] input)
        {
            int positions = InputShape.Height * InputShape.Width;
            trainedLast = mode.Training;
            normalized = new Quad[input.Length];
            inverseDeviations = new double[InputShape.Channels];
            var outputs = new Quad[input.Length];
            (double[] Mean, double[] Variance) record = recorded != null && trainedLast
                ? (new double[InputShape.Channels], new double[InputShape.Channels])
                : default;

            for (int c = 0; c < InputShape.Channels; c++)
            {
                int start = c * positions;
                double mean, variance;

                if (trainedLast)
                {
                    double sum = 0, squares = 0;
                    for (int i = start; i < start + positions; i++) sum += (float)input[i];
                    mean = sum / positions;
                    for (int i = start; i < start + positions; i++) squares += Math.Pow((float)input[i] - mean, 2);
                    variance = squares / positions;

                    double unbiased = positions > 1 ? squares / (positions - 1) : variance;
                    if (record.Mean != null)
                    {
                        record.Mean[c] = mean;
                        record.Variance[c] = unbiased;
                    }
                    else
                    {
                        UpdateRunning(c, mean, unbiased);
                    }
                }
                else
                {
                    mean = (float)RunningMean[c];
                    variance = (float)RunningVariance[c];
                }

                double inverse = 1 / Math.Sqrt(variance + Epsilon);
                inverseDeviations[c] = inverse;
                for (int i = start; i < start + positions; i++)
                {
                    normalized[i] = (Quad)(((float)input[i] - mean) * inverse);
                    outputs[i] = (Gamma.Values[c] * normalized[i]) + Beta.Values[c];
                }
            }

            if (record.Mean != null) recorded.Add(record);
            return activation.Apply(outputs);
        }

        public Quad[] Backpropagate(Quad[] outputGradient)
        {
            if (normalized == null) throw new InvalidOperationException("Propagate must be called before Backpropagate");

            Quad[] gradient = activation.Backpropagate(outputGradient);
            int positions = InputShape.Height * InputShape.Width;
            var inputGradient = new Quad[gradient.Length];
            var gammaGradients = new Quad[InputShape.Channels];
            var betaGradients = new Quad[InputShape.Channels];

            for (int c = 0; c < InputShape.Channels; c++)
            {
                int start = c * positions;
                double gamma = (float)Gamma.Values[c], inverse = inverseDeviations[c];
                double sumGradient = 0, sumGradientNormalized = 0;

                for (int i = start; i < start + positions; i++)
                {
                    sumGradient += (float)gradient[i];
                    sumGradientNormalized += (float)gradient[i] * (float)normalized[i];
                }

                gammaGradients[c] = (Quad)sumGradientNormalized;
                betaGradients[c] = (Quad)sumGradient;

                for (int i = start; i < start + positions; i++)
                {
                    double value = trainedLast
                        // the mean and variance depend on every input of the channel
                        ? gamma * inverse / positions * ((positions * (float)gradient[i]) - sumGradient - ((float)normalized[i] * sumGradientNormalized))
                        : gamma * inverse * (float)gradient[i];
                    inputGradient[i] = (Quad)value;
                }
            }

            Gamma.Update(gammaGradients);
            Beta.Update(betaGradients);
            return inputGradient;
        }
    }
}
