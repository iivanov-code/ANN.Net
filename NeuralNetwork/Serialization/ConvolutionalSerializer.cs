using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ANN.Net.Abstractions.Optimizers;
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
                        Weights = ToDocument(convolution.Weights),
                        Biases = ToDocument(convolution.Biases)
                    };

                case PoolingLayer pooling:
                    return new ConvolutionalLayerDocument
                    {
                        Type = ConvolutionalLayerType.Pooling,
                        Pooling = pooling.Type,
                        Size = pooling.Size,
                        Stride = pooling.Stride
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

                default:
                    throw new NotSupportedException($"Saving a {layer.GetType().Name} is not supported");
            }
        }

        private static ParametersDocument ToDocument(Parameters parameters)
        {
            float[] rates = parameters.LearningRates.ToArray();
            bool shared = rates.Length > 0 && rates.All(x => x.Equals(rates[0]));

            return new ParametersDocument
            {
                Values = parameters.Values.Select(x => (float)x).ToArray(),
                LearningRate = shared ? rates[0] : null,
                LearningRates = shared || rates.Length == 0 ? null : rates
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

            var network = new ConvolutionalNetwork(settings);
            for (int i = 0; i < layers.Count; i++)
            {
                switch (network.Layers[i])
                {
                    case ConvolutionLayer convolution:
                        Restore(convolution.Weights, layers[i].Weights, i);
                        Restore(convolution.Biases, layers[i].Biases, i);
                        break;

                    case DenseLayer dense:
                        Restore(dense.Weights, layers[i].Weights, i);
                        Restore(dense.Biases, layers[i].Biases, i);
                        break;
                }
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
                        ActivationType = Required(layer.Activation, nameof(layer.Activation))
                    };

                case ConvolutionalLayerType.Pooling:
                    return new PoolingLayerSettings
                    {
                        Type = Required(layer.Pooling, nameof(layer.Pooling)),
                        Size = Required(layer.Size, nameof(layer.Size)),
                        Stride = Required(layer.Stride, nameof(layer.Stride))
                    };

                case ConvolutionalLayerType.Dense:
                    return new DenseLayerSettings
                    {
                        NeuronsCount = Required(layer.NeuronsCount, nameof(layer.NeuronsCount)),
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

        private static void Restore(Parameters parameters, ParametersDocument document, int layer)
        {
            if (document?.Values == null || document.Values.Length != parameters.Values.Length)
                throw new InvalidDataException($"Layer {layer} needs {parameters.Values.Length} values");
            if (document.LearningRates != null && document.LearningRates.Length != parameters.Values.Length)
                throw new InvalidDataException($"Layer {layer} needs {parameters.Values.Length} learning rates");

            for (int i = 0; i < parameters.Values.Length; i++)
            {
                parameters.Values[i] = document.Values[i];

                float? rate = document.LearningRates?[i] ?? document.LearningRate;
                if (rate.HasValue && parameters.Optimizers[i] is LearningRateOptimizer optimizer)
                {
                    if (!(rate >= 0 && rate <= 1)) throw new InvalidDataException($"Invalid learning rate {rate}");
                    optimizer.LearningRate = rate.Value;
                }
            }
        }
    }
}
