using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;

namespace ANN.Net.Transformers
{
    /// <summary>
    /// Encoder-only transformer: input projection + positional encoding,
    /// a stack of <see cref="TransformerEncoderBlock"/>s, a final normalization and an output projection
    /// </summary>
    [Serializable]
    internal sealed class TransformerNetwork : ISequenceNetwork
    {
        private const double LOG_EPSILON = 1e-7;

        private readonly TransformerSettings settings;
        private readonly LinearProjection inputProjection;
        private readonly Quad[][] positionalEncodings;
        private readonly List<TransformerEncoderBlock> blocks;
        private readonly LayerNormalization outputNormalization;
        private readonly LinearProjection outputProjection;

        public TransformerNetwork(TransformerSettings settings)
        {
            if (settings == null) throw new ArgumentNullException(nameof(settings));
            Validate(settings);

            this.settings = settings;
            this.Optimizers = new List<LearningRateOptimizer>();

            this.inputProjection = new LinearProjection(settings.InputDimension,
                                                        settings.ModelDimension,
                                                        settings.OptimizerType,
                                                        settings.InitializationType,
                                                        this.Optimizers);

            this.positionalEncodings = PositionalEncoding.Build(settings.MaxSequenceLength, settings.ModelDimension);

            this.blocks = new List<TransformerEncoderBlock>(settings.LayersCount);
            for (int i = 0; i < settings.LayersCount; i++)
            {
                this.blocks.Add(new TransformerEncoderBlock(settings.ModelDimension,
                                                            settings.HeadsCount,
                                                            settings.FeedForwardDimension,
                                                            settings.UseCausalMask,
                                                            settings.FeedForwardActivationType,
                                                            settings.OptimizerType,
                                                            settings.InitializationType,
                                                            this.Optimizers));
            }

            this.outputNormalization = new LayerNormalization(settings.ModelDimension, settings.OptimizerType, this.Optimizers);

            this.outputProjection = new LinearProjection(settings.ModelDimension,
                                                         settings.OutputDimension,
                                                         settings.OptimizerType,
                                                         settings.InitializationType,
                                                         this.Optimizers);
        }

        public Quad[][] Values { get; private set; }

        public Quad[][] Errors { get; private set; }

        public ICollection<LearningRateOptimizer> Optimizers { get; }

        public Quad[][] Propagate(Quad[][] sequence)
        {
            if (sequence == null) throw new ArgumentNullException(nameof(sequence));
            if (sequence.Length < 1) throw new ArgumentException("Sequence must not be empty", nameof(sequence));
            if (sequence.Length > this.settings.MaxSequenceLength)
                throw new ArgumentException($"Sequence is longer than the maximum of {this.settings.MaxSequenceLength}", nameof(sequence));

            Quad[][] hidden = this.inputProjection.Propagate(sequence);
            for (int t = 0; t < hidden.Length; t++)
            {
                for (int i = 0; i < hidden[t].Length; i++)
                {
                    hidden[t][i] += this.positionalEncodings[t][i];
                }
            }

            foreach (TransformerEncoderBlock block in this.blocks)
            {
                hidden = block.Propagate(hidden);
            }

            hidden = this.outputNormalization.Propagate(hidden);
            Quad[][] outputs = this.outputProjection.Propagate(hidden);

            if (this.settings.OutputApplySoftMax)
            {
                for (int t = 0; t < outputs.Length; t++)
                {
                    TransformerUtils.SoftMax(outputs[t]);
                }
            }

            this.Values = outputs;
            return outputs;
        }

        public Quad Backpropagate(Quad[][] targets)
        {
            if (targets == null) throw new ArgumentNullException(nameof(targets));
            if (this.Values == null) throw new InvalidOperationException("Propagate must be called before Backpropagate");
            if (targets.Length != this.Values.Length)
                throw new ArgumentException("Targets sequence length differs from the propagated sequence length", nameof(targets));

            Quad totalError = 0;
            Quad[][] errors = new Quad[targets.Length][];

            for (int t = 0; t < targets.Length; t++)
            {
                if (targets[t].Length != this.settings.OutputDimension)
                    throw new ArgumentException($"Expected targets of size {this.settings.OutputDimension} but found {targets[t].Length}", nameof(targets));

                errors[t] = new Quad[targets[t].Length];
                for (int i = 0; i < targets[t].Length; i++)
                {
                    Quad predicted = this.Values[t][i];
                    Quad target = targets[t][i];

                    // (predicted - target) is the gradient of both the squared error
                    // on a linear output and the cross entropy on a soft-max output
                    errors[t][i] = predicted - target;
                    totalError += this.CalculateLoss(target, predicted);
                }
            }

            this.Errors = errors;

            Quad[][] gradients = this.outputProjection.Backpropagate(errors);
            gradients = this.outputNormalization.Backpropagate(gradients);

            for (int i = this.blocks.Count - 1; i >= 0; i--)
            {
                gradients = this.blocks[i].Backpropagate(gradients);
            }

            // positional encodings are fixed, only the input projection learns
            this.inputProjection.Backpropagate(gradients);

            return totalError;
        }

        private Quad CalculateLoss(Quad target, Quad predicted)
        {
            if (this.settings.OutputApplySoftMax)
            {
                return (Quad)(-target * Math.Log(Math.Max(predicted, LOG_EPSILON)));
            }

            Quad difference = predicted - target;
            return difference * difference / 2;
        }

        private static void Validate(TransformerSettings settings)
        {
            if (settings.InputDimension < 1) throw new ArgumentException("Must be greater than 0", nameof(settings.InputDimension));
            if (settings.ModelDimension < 1) throw new ArgumentException("Must be greater than 0", nameof(settings.ModelDimension));
            if (settings.HeadsCount < 1) throw new ArgumentException("Must be greater than 0", nameof(settings.HeadsCount));
            if (settings.FeedForwardDimension < 1) throw new ArgumentException("Must be greater than 0", nameof(settings.FeedForwardDimension));
            if (settings.LayersCount < 1) throw new ArgumentException("Must be greater than 0", nameof(settings.LayersCount));
            if (settings.OutputDimension < 1) throw new ArgumentException("Must be greater than 0", nameof(settings.OutputDimension));
            if (settings.MaxSequenceLength < 1) throw new ArgumentException("Must be greater than 0", nameof(settings.MaxSequenceLength));
            if (settings.ModelDimension % settings.HeadsCount != 0)
                throw new ArgumentException("Model dimension must be divisible by the heads count", nameof(settings.HeadsCount));
        }
    }
}
