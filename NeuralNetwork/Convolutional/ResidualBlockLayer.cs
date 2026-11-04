using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Optimizers;

namespace ANN.Net.Convolutional
{
    /// <summary>
    /// Basic residual block of ResNet-18/34: activation(main(x) + shortcut(x)) with
    /// main = conv 3x3 (stride) → batch norm → activation → conv 3x3 → batch norm, and shortcut = conv 1x1 (stride) →
    /// batch norm when the stride or the channels change, otherwise the input itself
    /// </summary>
    [Serializable]
    internal sealed class ResidualBlockLayer : IConvolutionalLayer
    {
        private const float Momentum = 0.1f;
        private const float Epsilon = 1e-5f;

        private readonly Activation activation;

        public ResidualBlockLayer(Shape inputShape, int filters, int stride, ActivationTypes activationType, NetworkMode mode,
                                  OptimizerTypes optimizerType, WeightInitType initializationType, ICollection<LearningRateOptimizer> optimizers)
        {
            if (filters < 1) throw new ArgumentOutOfRangeException(nameof(filters), filters, "Must be at least 1");

            var first = new ConvolutionLayer(inputShape, filters, 3, 3, stride, 1, ActivationTypes.Identity, optimizerType, initializationType, optimizers, useBias: false);
            var firstNormalization = new BatchNormalizationLayer(first.OutputShape, activationType, Momentum, Epsilon, mode, optimizerType, optimizers);
            var second = new ConvolutionLayer(first.OutputShape, filters, 3, 3, 1, 1, ActivationTypes.Identity, optimizerType, initializationType, optimizers, useBias: false);
            var secondNormalization = new BatchNormalizationLayer(second.OutputShape, ActivationTypes.Identity, Momentum, Epsilon, mode, optimizerType, optimizers);
            Main = new IConvolutionalLayer[] { first, firstNormalization, second, secondNormalization };

            if (stride != 1 || inputShape.Channels != filters)
            {
                var projection = new ConvolutionLayer(inputShape, filters, 1, 1, stride, 0, ActivationTypes.Identity, optimizerType, initializationType, optimizers, useBias: false);
                Shortcut = new IConvolutionalLayer[] { projection, new BatchNormalizationLayer(projection.OutputShape, ActivationTypes.Identity, Momentum, Epsilon, mode, optimizerType, optimizers) };
            }
            else
            {
                Shortcut = Array.Empty<IConvolutionalLayer>();
            }

            InputShape = inputShape;
            OutputShape = secondNormalization.OutputShape;
            Filters = filters;
            Stride = stride;
            activation = new Activation(activationType);

            Shape shortcutShape = Shortcut.Count > 0 ? Shortcut[^1].OutputShape : inputShape;
            if (shortcutShape != OutputShape)
                throw new ArgumentException($"The shortcut gives {shortcutShape} but the block gives {OutputShape}");
        }

        private ResidualBlockLayer(ResidualBlockLayer master, ReplicaContext context)
        {
            Main = master.Main.Select(x => x.Replicate(context)).ToArray();
            Shortcut = master.Shortcut.Select(x => x.Replicate(context)).ToArray();
            InputShape = master.InputShape;
            OutputShape = master.OutputShape;
            Filters = master.Filters;
            Stride = master.Stride;
            activation = new Activation(master.ActivationType);
        }

        public IConvolutionalLayer Replicate(ReplicaContext context) => new ResidualBlockLayer(this, context);

        public Shape InputShape { get; }

        public Shape OutputShape { get; }

        public int Filters { get; }

        public int Stride { get; }

        public ActivationTypes ActivationType => activation.Type;

        public IReadOnlyList<IConvolutionalLayer> Main { get; }

        /// <summary>
        /// Projection of the input; empty when the input is added unchanged
        /// </summary>
        public IReadOnlyList<IConvolutionalLayer> Shortcut { get; }

        public Quad[] Propagate(Quad[] input)
        {
            Quad[] main = Main.Aggregate(input, (value, layer) => layer.Propagate(value));
            Quad[] shortcut = Shortcut.Aggregate(input, (value, layer) => layer.Propagate(value));

            var sum = new Quad[main.Length];
            for (int i = 0; i < sum.Length; i++) sum[i] = main[i] + shortcut[i];
            return activation.Apply(sum);
        }

        public Quad[] Backpropagate(Quad[] outputGradient)
        {
            Quad[] gradient = activation.Backpropagate(outputGradient);
            Quad[] main = Main.Reverse().Aggregate(gradient, (value, layer) => layer.Backpropagate(value));
            Quad[] shortcut = Shortcut.Reverse().Aggregate(gradient, (value, layer) => layer.Backpropagate(value));

            var inputGradient = new Quad[main.Length];
            for (int i = 0; i < inputGradient.Length; i++) inputGradient[i] = main[i] + shortcut[i];
            return inputGradient;
        }
    }
}
