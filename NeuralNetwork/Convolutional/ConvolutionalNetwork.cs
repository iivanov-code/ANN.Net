using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Transformers;

namespace ANN.Net.Convolutional
{
    /// <summary>
    /// Convolutional network: convolution, pooling and dense layers followed by a dense output layer. Like the
    /// transformer it works on whole arrays, and every weight and bias array has an optimizer that is applied as soon as its
    /// gradient is known
    /// </summary>
    [Serializable]
    internal sealed class ConvolutionalNetwork : ANNetwork, IConvolutionalNetwork
    {
        private const double LOG_EPSILON = 1e-7;

        private readonly NetworkMode mode = new NetworkMode();

        public ConvolutionalNetwork(ConvolutionalNetworkSettings settings)
        {
            ArgumentNullException.ThrowIfNull(settings);
            if (settings.InputChannels < 1 || settings.InputHeight < 1 || settings.InputWidth < 1)
                throw new ArgumentException("The input needs at least one channel, row and column", nameof(settings));
            if (settings.OutputDimension < 1)
                throw new ArgumentException("The network needs at least one output", nameof(settings));

            Settings = settings;
            Optimizers = new List<LearningRateOptimizer>();
            using IDisposable seed = Utils.NetworkUtils.SeedRandom(settings.Seed);

            var shape = new Shape(settings.InputChannels, settings.InputHeight, settings.InputWidth);
            bool dense = false;
            var layers = new List<IConvolutionalLayer>();

            foreach (ConvolutionalLayerSettings layerSettings in settings.Layers ?? Enumerable.Empty<ConvolutionalLayerSettings>())
            {
                IConvolutionalLayer layer;
                switch (layerSettings)
                {
                    case ConvolutionLayerSettings convolution:
                        if (dense) throw new ArgumentException("Convolution layers must come before dense layers", nameof(settings));
                        layer = new ConvolutionLayer(shape, convolution.Filters, convolution.KernelHeight, convolution.KernelWidth, convolution.Stride,
                                                     convolution.Padding, convolution.ActivationType, settings.OptimizerType, settings.InitializationType, Optimizers,
                                                     convolution.UseBias);
                        break;

                    case PoolingLayerSettings pooling:
                        if (dense) throw new ArgumentException("Pooling layers must come before dense layers", nameof(settings));
                        layer = new PoolingLayer(shape, pooling.Type, pooling.Size, pooling.Stride == 0 ? pooling.Size : pooling.Stride, pooling.Padding);
                        break;

                    case BatchNormalizationLayerSettings normalization:
                        if (dense) throw new ArgumentException("Batch normalization layers must come before dense layers", nameof(settings));
                        layer = new BatchNormalizationLayer(shape, normalization.ActivationType, normalization.Momentum, normalization.Epsilon, mode,
                                                            settings.OptimizerType, Optimizers);
                        break;

                    case GlobalAveragePoolingLayerSettings:
                        if (dense) throw new ArgumentException("Global average pooling must come before dense layers", nameof(settings));
                        layer = new GlobalAveragePoolingLayer(shape);
                        break;

                    case ResidualBlockSettings residual:
                        if (dense) throw new ArgumentException("Residual blocks must come before dense layers", nameof(settings));
                        layer = new ResidualBlockLayer(shape, residual.Filters, residual.Stride, residual.ActivationType, mode,
                                                       settings.OptimizerType, settings.InitializationType, Optimizers);
                        break;

                    case DenseLayerSettings denseSettings:
                        dense = true;
                        layer = new DenseLayer(shape, denseSettings.NeuronsCount, denseSettings.ActivationType, settings.OptimizerType, settings.InitializationType, Optimizers);
                        break;

                    default:
                        throw new ArgumentException($"Unknown layer {layerSettings?.GetType().Name ?? "null"}", nameof(settings));
                }

                layers.Add(layer);
                shape = layer.OutputShape;
            }

            ActivationTypes outputActivation = settings.OutputApplySoftMax ? ActivationTypes.Identity : settings.OutputActivationType;
            layers.Add(new DenseLayer(shape, settings.OutputDimension, outputActivation, settings.OptimizerType, settings.InitializationType, Optimizers));
            Layers = layers;
            Errors = new Quad[settings.OutputDimension];
        }

        internal ConvolutionalNetworkSettings Settings { get; }

        /// <summary>
        /// All layers including the dense output layer
        /// </summary>
        internal IReadOnlyList<IConvolutionalLayer> Layers { get; }

        public ICollection<LearningRateOptimizer> Optimizers { get; }

        public long ParameterCount => Optimizers.OfType<Optimizers.ArrayOptimizer>().Sum(x => (long)x.Length);

        public int InputChannels => Settings.InputChannels;

        public int InputHeight => Settings.InputHeight;

        public int InputWidth => Settings.InputWidth;

        public int OutputDimension => Settings.OutputDimension;

        public bool OutputApplySoftMax => Settings.OutputApplySoftMax;

        public bool Training
        {
            get => mode.Training;
            set => mode.Training = value;
        }

        public Quad[] Values { get; private set; }

        public Quad[] Errors { get; }

        public Quad[] InputGradients { get; private set; } = Array.Empty<Quad>();

        public IReadOnlyList<Tuple<Quad, Quad>> MinMaxValues => null;

        public Quad[] Propagate(params Quad[] values)
        {
            ArgumentNullException.ThrowIfNull(values);
            int size = InputChannels * InputHeight * InputWidth;
            if (values.Length != size)
                throw new ArgumentException($"Expected {size} input values ({InputChannels}x{InputHeight}x{InputWidth}) but found {values.Length}", nameof(values));

            Quad[] current = values;
            foreach (IConvolutionalLayer layer in Layers)
            {
                current = layer.Propagate(current);
            }

            if (OutputApplySoftMax)
            {
                current = (Quad[])current.Clone();
                TransformerUtils.SoftMax(current);
            }

            Values = current;
            return current;
        }

        /// <summary>
        /// Trains on the image of the last propagation. Returns the cross entropy for soft-max outputs, otherwise half
        /// the squared error
        /// </summary>
        public Quad Backpropagate(params Quad[] targets)
        {
            ArgumentNullException.ThrowIfNull(targets);
            if (Values == null) throw new InvalidOperationException("Propagate must be called before Backpropagate");
            if (targets.Length != OutputDimension)
                throw new ArgumentException($"Expected {OutputDimension} targets but found {targets.Length}", nameof(targets));

            double loss = 0;
            var gradient = new Quad[targets.Length];
            for (int i = 0; i < targets.Length; i++)
            {
                Quad error = Values[i] - targets[i];
                Errors[i] = error;

                // (predicted - target) is the gradient of half the squared error with respect to the outputs, and of
                // the cross entropy with respect to the values before the soft-max
                gradient[i] = error;
                loss += OutputApplySoftMax
                    ? -(float)targets[i] * Math.Log(Math.Max((float)Values[i], LOG_EPSILON))
                    : 0.5 * (float)error * (float)error;
            }

            for (int i = Layers.Count - 1; i >= 0; i--)
            {
                gradient = Layers[i].Backpropagate(gradient);
            }

            InputGradients = gradient;
            return (Quad)loss;
        }
    }
}
