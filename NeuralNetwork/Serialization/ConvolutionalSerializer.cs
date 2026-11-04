using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Convolutional;

namespace ANN.Net.Serialization
{
    /// <summary>
    /// Converts convolutional networks to and from <see cref="ConvolutionalDocument"/>
    /// </summary>
    internal static class ConvolutionalSerializer
    {
        public static ConvolutionalDocument ToDocument(ConvolutionalNetwork network)
        {
            ConvolutionalNetworkSettings settings = network.Settings;
            return new ConvolutionalDocument
            {
                InputChannels = settings.InputChannels,
                InputHeight = settings.InputHeight,
                InputWidth = settings.InputWidth,
                OutputDimension = settings.OutputDimension,
                OutputApplySoftMax = settings.OutputApplySoftMax,
                OutputActivationType = settings.OutputActivationType,
                OptimizerType = settings.OptimizerType,
                Training = network.Training,
                Layers = network.Layers.Select(ToDocument).ToList()
            };
        }

        private static ConvolutionalLayerDocument ToDocument(IConvolutionalLayer layer)
        {
            switch (layer)
            {
                case ConvolutionLayer convolution:
                    return new ConvolutionalLayerDocument
                    {
                        Type = ConvolutionalLayerType.Convolution,
                        Activation = convolution.ActivationType,
                        Filters = convolution.OutputShape.Channels,
                        KernelHeight = convolution.KernelHeight,
                        KernelWidth = convolution.KernelWidth,
                        Stride = convolution.Stride,
                        Padding = convolution.Padding,
                        UseBias = convolution.Biases != null,
                        Weights = ToDocument(convolution.Weights),
                        Biases = convolution.Biases == null ? null : ToDocument(convolution.Biases)
                    };

                case PoolingLayer pooling:
                    return new ConvolutionalLayerDocument
                    {
                        Type = ConvolutionalLayerType.Pooling,
                        Pooling = pooling.Type,
                        Size = pooling.Size,
                        Stride = pooling.Stride,
                        Padding = pooling.Padding
                    };

                case DenseLayer dense:
                    return new ConvolutionalLayerDocument
                    {
                        Type = ConvolutionalLayerType.Dense,
                        Activation = dense.ActivationType,
                        NeuronsCount = dense.OutputShape.Size,
                        Weights = ToDocument(dense.Weights),
                        Biases = ToDocument(dense.Biases)
                    };

                case BatchNormalizationLayer normalization:
                    return new ConvolutionalLayerDocument
                    {
                        Type = ConvolutionalLayerType.BatchNormalization,
                        Activation = normalization.ActivationType,
                        Momentum = normalization.Momentum,
                        Epsilon = normalization.Epsilon,
                        Weights = ToDocument(normalization.Gamma),
                        Biases = ToDocument(normalization.Beta),
                        RunningMean = normalization.RunningMean.Select(x => (float)x).ToArray(),
                        RunningVariance = normalization.RunningVariance.Select(x => (float)x).ToArray()
                    };

                case GlobalAveragePoolingLayer:
                    return new ConvolutionalLayerDocument { Type = ConvolutionalLayerType.GlobalAveragePooling };

                case ResidualBlockLayer residual:
                    return new ConvolutionalLayerDocument
                    {
                        Type = ConvolutionalLayerType.Residual,
                        Activation = residual.ActivationType,
                        Filters = residual.Filters,
                        Stride = residual.Stride,
                        Main = residual.Main.Select(ToDocument).ToList(),
                        Shortcut = residual.Shortcut.Select(ToDocument).ToList()
                    };

                default:
                    throw new NotSupportedException($"Saving a {layer.GetType().Name} is not supported");
            }
        }

        private static ParametersDocument ToDocument(Parameters parameters)
        {
            return new ParametersDocument
            {
                Values = parameters.Values.Select(x => (float)x).ToArray(),
                LearningRate = parameters.LearningRate
            };
        }

        public static ConvolutionalNetwork FromDocument(ConvolutionalDocument document)
        {
            if (document == null) throw new InvalidDataException("The file holds no convolutional layers");
            List<ConvolutionalLayerDocument> layers = document.Layers ?? new List<ConvolutionalLayerDocument>();
            if (layers.Count == 0 || layers[^1]?.Type != ConvolutionalLayerType.Dense)
                throw new InvalidDataException("A convolutional network ends with its dense output layer");

            var settings = new ConvolutionalNetworkSettings
            {
                InputChannels = document.InputChannels,
                InputHeight = document.InputHeight,
                InputWidth = document.InputWidth,
                OutputDimension = document.OutputDimension,
                OutputApplySoftMax = document.OutputApplySoftMax,
                OutputActivationType = document.OutputActivationType,
                OptimizerType = document.OptimizerType
            };

            foreach (ConvolutionalLayerDocument layer in layers.Take(layers.Count - 1))
            {
                settings.Layers.Add(ToSettings(layer));
            }

            var network = new ConvolutionalNetwork(settings) { Training = document.Training };
            for (int i = 0; i < layers.Count; i++)
            {
                Restore(network.Layers[i], layers[i], i.ToString(System.Globalization.CultureInfo.InvariantCulture));
            }

            return network;
        }

        private static ConvolutionalLayerSettings ToSettings(ConvolutionalLayerDocument layer)
        {
            switch (layer?.Type)
            {
                case ConvolutionalLayerType.Convolution:
                    return new ConvolutionLayerSettings
                    {
                        Filters = Required(layer.Filters, nameof(layer.Filters)),
                        KernelHeight = Required(layer.KernelHeight, nameof(layer.KernelHeight)),
                        KernelWidth = Required(layer.KernelWidth, nameof(layer.KernelWidth)),
                        Stride = Required(layer.Stride, nameof(layer.Stride)),
                        Padding = Required(layer.Padding, nameof(layer.Padding)),
                        ActivationType = Required(layer.Activation, nameof(layer.Activation)),
                        UseBias = layer.UseBias ?? true
                    };

                case ConvolutionalLayerType.Pooling:
                    return new PoolingLayerSettings
                    {
                        Type = Required(layer.Pooling, nameof(layer.Pooling)),
                        Size = Required(layer.Size, nameof(layer.Size)),
                        Stride = Required(layer.Stride, nameof(layer.Stride)),
                        Padding = layer.Padding ?? 0
                    };

                case ConvolutionalLayerType.Dense:
                    return new DenseLayerSettings
                    {
                        NeuronsCount = Required(layer.NeuronsCount, nameof(layer.NeuronsCount)),
                        ActivationType = Required(layer.Activation, nameof(layer.Activation))
                    };

                case ConvolutionalLayerType.BatchNormalization:
                    return new BatchNormalizationLayerSettings
                    {
                        ActivationType = Required(layer.Activation, nameof(layer.Activation)),
                        Momentum = Required(layer.Momentum, nameof(layer.Momentum)),
                        Epsilon = Required(layer.Epsilon, nameof(layer.Epsilon))
                    };

                case ConvolutionalLayerType.GlobalAveragePooling:
                    return new GlobalAveragePoolingLayerSettings();

                case ConvolutionalLayerType.Residual:
                    return new ResidualBlockSettings
                    {
                        Filters = Required(layer.Filters, nameof(layer.Filters)),
                        Stride = Required(layer.Stride, nameof(layer.Stride)),
                        ActivationType = Required(layer.Activation, nameof(layer.Activation))
                    };

                default:
                    throw new InvalidDataException("Unknown convolutional layer");
            }
        }

        private static T Required<T>(T? value, string name)
            where T : struct
        {
            return value ?? throw new InvalidDataException($"A convolutional layer is missing {name}");
        }

        /// <summary>
        /// Copies the saved values into a layer built from the same settings
        /// </summary>
        private static void Restore(IConvolutionalLayer layer, ConvolutionalLayerDocument document, string path)
        {
            switch (layer)
            {
                case ConvolutionLayer convolution:
                    Restore(convolution.Weights, document.Weights, path);
                    if (convolution.Biases != null) Restore(convolution.Biases, document.Biases, path);
                    break;

                case DenseLayer dense:
                    Restore(dense.Weights, document.Weights, path);
                    Restore(dense.Biases, document.Biases, path);
                    break;

                case BatchNormalizationLayer normalization:
                    Restore(normalization.Gamma, document.Weights, path);
                    Restore(normalization.Beta, document.Biases, path);
                    Restore(normalization.RunningMean, document.RunningMean, path);
                    Restore(normalization.RunningVariance, document.RunningVariance, path);
                    break;

                case ResidualBlockLayer residual:
                    if (document.Main?.Count != residual.Main.Count || (document.Shortcut?.Count ?? 0) != residual.Shortcut.Count)
                        throw new InvalidDataException($"Residual block {path} does not have the saved layers");

                    for (int i = 0; i < residual.Main.Count; i++) Restore(residual.Main[i], document.Main[i], $"{path}.main{i}");
                    for (int i = 0; i < residual.Shortcut.Count; i++) Restore(residual.Shortcut[i], document.Shortcut[i], $"{path}.shortcut{i}");
                    break;
            }
        }

        private static void Restore(Quad[] values, float[] saved, string path)
        {
            if (saved == null || saved.Length != values.Length)
                throw new InvalidDataException($"Layer {path} needs {values.Length} running statistics");

            for (int i = 0; i < values.Length; i++) values[i] = saved[i];
        }

        private static void Restore(Parameters parameters, ParametersDocument document, string layer)
        {
            if (document?.Values == null || document.Values.Length != parameters.Values.Length)
                throw new InvalidDataException($"Layer {layer} needs {parameters.Values.Length} values");
            if (document.LearningRates != null && document.LearningRates.Length != parameters.Values.Length)
                throw new InvalidDataException($"Layer {layer} needs {parameters.Values.Length} learning rates");

            for (int i = 0; i < parameters.Values.Length; i++)
            {
                parameters.Values[i] = document.Values[i];
            }

            // files written before the optimizers kept their state in arrays may hold one rate per value
            float? rate = document.LearningRate ?? document.LearningRates?.FirstOrDefault();
            if (document.LearningRates != null && document.LearningRates.Any(x => !x.Equals(rate)))
                throw new InvalidDataException($"Layer {layer} holds different learning rates per value; all values of a layer share one now");

            if (rate.HasValue)
            {
                if (!(rate >= 0 && rate <= 1)) throw new InvalidDataException($"Invalid learning rate {rate}");
                parameters.LearningRate = rate.Value;
            }
        }
    }
}
