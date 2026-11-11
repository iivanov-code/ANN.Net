using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Optimizers;
using ANN.Net.Transformers;
using static ANN.Net.Recurrent.RecurrentMath;

namespace ANN.Net.Recurrent
{
    /// <summary>
    /// Stack of recurrent layers with a linear output at every step: y(t) = W h(t) + b of the last layer, through a
    /// soft-max when <see cref="RecurrentNetworkSettings.OutputApplySoftMax"/> is set. A sequence trains through all of
    /// its steps (backpropagation through time); the gradient of all weights is collected for the whole sequence,
    /// clipped, and applied once
    /// </summary>
    internal sealed class RecurrentNetwork : IRecurrentNetwork
    {
        private const double LogEpsilon = 1e-7;

        private readonly RecurrentNetworkSettings settings;
        private readonly RecurrentLayer[] layers;
        private readonly RecurrentTensor outputWeights;
        private readonly RecurrentTensor outputBiases;
        private readonly RecurrentTensor[] tensors;
        private float[] topHidden = Array.Empty<float>();
        private int steps;
        private float[][] state;
        private BatchTrainer<RecurrentNetwork> trainer;

        public RecurrentNetwork(RecurrentNetworkSettings settings)
        {
            ArgumentNullException.ThrowIfNull(settings);
            Validate(settings);

            this.settings = settings;
            Optimizers = new List<LearningRateOptimizer>();
            using IDisposable seed = Utils.NetworkUtils.SeedRandom(settings.Seed);

            layers = new RecurrentLayer[settings.LayersCount];
            for (int l = 0; l < layers.Length; l++)
            {
                int inputs = l == 0 ? settings.InputDimension : settings.HiddenDimension;
                layers[l] = CreateLayer(settings, inputs, Optimizers);
                layers[l].Initialize(settings.InitializationType);
            }

            outputWeights = new RecurrentTensor(settings.HiddenDimension * settings.OutputDimension, settings.OptimizerType, Optimizers);
            outputBiases = new RecurrentTensor(settings.OutputDimension, settings.OptimizerType, Optimizers);
            Span<float> weights = outputWeights.Weights;
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = TransformerUtils.InitialWeight(settings.InitializationType, settings.HiddenDimension, settings.OutputDimension);
            }

            tensors = Collect();
        }

        private RecurrentNetwork(RecurrentNetwork master, ReplicaContext context)
        {
            settings = master.settings;
            Optimizers = new List<LearningRateOptimizer>();
            layers = master.layers.Select(x => x.Replicate(context)).ToArray();
            outputWeights = master.outputWeights.Replicate(context);
            outputBiases = master.outputBiases.Replicate(context);
            tensors = Collect();
        }

        public NeuronType CellType => settings.CellType;

        public int HiddenDimension => settings.HiddenDimension;

        public int LayersCount => layers.Length;

        public long ParameterCount => tensors.Sum(x => (long)x.Length);

        public Quad[][] Values { get; private set; }

        public Quad[][] Errors { get; private set; }

        public ICollection<LearningRateOptimizer> Optimizers { get; }

        public int InputDimension => settings.InputDimension;

        public int OutputDimension => settings.OutputDimension;

        /// <summary>
        /// A recurrent network takes sequences of any length
        /// </summary>
        public int MaxSequenceLength => int.MaxValue;

        public bool OutputApplySoftMax => settings.OutputApplySoftMax;

        internal IReadOnlyList<RecurrentLayer> Layers => layers;

        internal RecurrentTensor OutputWeights => outputWeights;

        internal RecurrentTensor OutputBiases => outputBiases;

        internal IReadOnlyList<RecurrentTensor> Tensors => tensors;

        public Quad[][] Propagate(Quad[][] sequence)
        {
            ArgumentNullException.ThrowIfNull(sequence);
            if (sequence.Length < 1) throw new ArgumentException("Sequence must not be empty", nameof(sequence));

            int inputs = settings.InputDimension;
            var flat = new float[sequence.Length * inputs];
            for (int t = 0; t < sequence.Length; t++)
            {
                Quad[] vector = sequence[t] ?? throw new ArgumentException($"Step {t} of the sequence is null", nameof(sequence));
                if (vector.Length != inputs) throw new ArgumentException($"Expected vectors of size {inputs} but found {vector.Length}", nameof(sequence));
                for (int i = 0; i < inputs; i++) flat[(t * inputs) + i] = vector[i];
            }

            steps = sequence.Length;
            ReadOnlySpan<float> hidden = flat;
            foreach (RecurrentLayer layer in layers) hidden = layer.Propagate(hidden, steps);

            if (topHidden.Length < hidden.Length) topHidden = new float[hidden.Length];
            hidden.CopyTo(topHidden);

            var outputs = new Quad[steps][];
            for (int t = 0; t < steps; t++)
            {
                outputs[t] = Output(hidden.Slice(t * settings.HiddenDimension, settings.HiddenDimension));
            }

            Values = outputs;
            return outputs;
        }

        public Quad Backpropagate(Quad[][] targets)
        {
            ArgumentNullException.ThrowIfNull(targets);
            if (Values == null) throw new InvalidOperationException("Propagate must be called before Backpropagate");
            if (targets.Length != Values.Length)
                throw new ArgumentException("Targets sequence length differs from the propagated sequence length", nameof(targets));

            Quad total = 0;
            var errors = new Quad[steps][];
            for (int t = 0; t < steps; t++)
            {
                errors[t] = new Quad[settings.OutputDimension];
                if (targets[t] == null) continue;
                if (targets[t].Length != settings.OutputDimension)
                    throw new ArgumentException($"Expected targets of size {settings.OutputDimension} but found {targets[t].Length}", nameof(targets));

                for (int i = 0; i < settings.OutputDimension; i++)
                {
                    Quad predicted = Values[t][i], target = targets[t][i];

                    // (predicted - target) is the gradient of both the squared error of a linear output and the cross
                    // entropy of a soft-max output
                    errors[t][i] = predicted - target;
                    total += Loss(target, predicted);
                }
            }

            Errors = errors;
            Train(errors);
            return total;
        }

        /// <summary>
        /// One step of mini-batch gradient descent on several threads; see <see cref="ISequenceNetwork.TrainBatch"/>.
        /// Every sequence is clipped on its own before the gradients are averaged
        /// </summary>
        public Quad TrainBatch(IReadOnlyList<Quad[][]> sequences, IReadOnlyList<Quad[][]> targets, BatchOptions options = null)
        {
            ArgumentNullException.ThrowIfNull(sequences);
            ArgumentNullException.ThrowIfNull(targets);
            if (sequences.Count != targets.Count) throw new ArgumentException("Every sequence needs targets", nameof(targets));

            trainer ??= new BatchTrainer<RecurrentNetwork>(context => new RecurrentNetwork(this, context));
            return trainer.Train(sequences.Count, options, (replica, i) =>
            {
                replica.Propagate(sequences[i]);
                return replica.Backpropagate(targets[i]);
            });
        }

        public Quad[] Step(Quad[] input)
        {
            ArgumentNullException.ThrowIfNull(input);
            if (input.Length != settings.InputDimension)
                throw new ArgumentException($"Expected {settings.InputDimension} inputs but found {input.Length}", nameof(input));

            state ??= layers.Select(x => new float[x.StateSize]).ToArray();
            ReadOnlySpan<float> x = input.Select(v => (float)v).ToArray();
            for (int l = 0; l < layers.Length; l++)
            {
                RecurrentLayer layer = layers[l];
                var next = new float[layer.StateSize];
                layer.Step(x, state[l], next, new float[layer.ScratchSize]);
                state[l] = next;
                x = next.AsSpan(0, layer.HiddenSize);
            }

            return Output(x);
        }

        public void ResetState() => state = null;

        private Quad[] Output(ReadOnlySpan<float> hidden)
        {
            int outputs = settings.OutputDimension;
            var sums = new float[outputs];
            outputBiases.Weights.CopyTo(sums);
            Affine(hidden, outputWeights.Weights, 0, outputs, 0, sums);

            var output = new Quad[outputs];
            for (int i = 0; i < outputs; i++) output[i] = sums[i];
            if (settings.OutputApplySoftMax) TransformerUtils.SoftMax(output);
            return output;
        }

        /// <summary>
        /// Back through the output and every layer with the gradient of every output vector; then the collected
        /// gradient is clipped and applied
        /// </summary>
        private void Train(Quad[][] errors)
        {
            int hiddenSize = settings.HiddenDimension, outputs = settings.OutputDimension;
            var hiddenGradients = new float[steps * hiddenSize];
            var deltas = new float[outputs];
            for (int t = 0; t < steps; t++)
            {
                for (int i = 0; i < outputs; i++) deltas[i] = errors[t][i];
                Axpy(1, deltas, outputBiases.Gradients);
                AffineBackward(topHidden.AsSpan(t * hiddenSize, hiddenSize), outputWeights.Weights, outputWeights.Gradients, 0, outputs, 0,
                               deltas, hiddenGradients.AsSpan(t * hiddenSize, hiddenSize));
            }

            float[] gradients = hiddenGradients;
            for (int l = layers.Length - 1; l >= 0; l--) gradients = layers[l].Backpropagate(gradients);

            if (settings.GradientClipping > 0)
            {
                double norm = Math.Sqrt(tensors.Sum(x => x.SquaredGradientNorm()));
                if (norm > settings.GradientClipping)
                {
                    float factor = (float)(settings.GradientClipping / norm);
                    foreach (RecurrentTensor tensor in tensors) tensor.ScaleGradients(factor);
                }
            }

            foreach (RecurrentTensor tensor in tensors) tensor.Apply();
        }

        private Quad Loss(Quad target, Quad predicted)
        {
            if (settings.OutputApplySoftMax) return (Quad)(-target * Math.Log(Math.Max(predicted, LogEpsilon)));

            Quad difference = predicted - target;
            return difference * difference / 2;
        }

        private RecurrentTensor[] Collect() =>
            layers.SelectMany(x => x.Tensors).Append(outputWeights).Append(outputBiases).ToArray();

        private static RecurrentLayer CreateLayer(RecurrentNetworkSettings settings, int inputs, ICollection<LearningRateOptimizer> optimizers)
        {
            return settings.CellType switch
            {
                NeuronType.LSTM => new LstmLayer(inputs, settings.HiddenDimension, false, settings.ForgetGateBias, settings.OptimizerType, optimizers),
                NeuronType.LSTMPeepHole => new LstmLayer(inputs, settings.HiddenDimension, true, settings.ForgetGateBias, settings.OptimizerType, optimizers),
                NeuronType.GRU => new GruLayer(inputs, settings.HiddenDimension, settings.ForgetGateBias, settings.OptimizerType, optimizers),
                NeuronType.RNN => new SimpleRecurrentLayer(inputs, settings.HiddenDimension, settings.OptimizerType, optimizers),
                _ => throw new ArgumentException($"{settings.CellType} is not a recurrent cell", nameof(settings)),
            };
        }

        private static void Validate(RecurrentNetworkSettings settings)
        {
            if (settings.InputDimension < 1) throw new ArgumentException("Must be greater than 0", nameof(settings.InputDimension));
            if (settings.HiddenDimension < 1) throw new ArgumentException("Must be greater than 0", nameof(settings.HiddenDimension));
            if (settings.OutputDimension < 1) throw new ArgumentException("Must be greater than 0", nameof(settings.OutputDimension));
            if (settings.LayersCount < 1) throw new ArgumentException("Must be greater than 0", nameof(settings.LayersCount));
            if (!(settings.GradientClipping >= 0) || float.IsInfinity(settings.GradientClipping))
                throw new ArgumentException("Must be 0 or a positive number", nameof(settings.GradientClipping));
            if (!float.IsFinite(settings.ForgetGateBias)) throw new ArgumentException("Must be a finite number", nameof(settings.ForgetGateBias));
        }
    }
}
