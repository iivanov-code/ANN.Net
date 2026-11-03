using System;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Convolutional;

namespace ANN.Net.Onnx
{
    /// <summary>
    /// ONNX graph of a convolutional network: the flat input is reshaped to [batch, channels, height, width], then
    /// Conv, BatchNormalization, MaxPool / AveragePool / GlobalAveragePool, residual Add, Flatten and MatMul + Add with
    /// their activations follow, and Softmax for soft-max outputs. Batch normalization uses its running averages, so the
    /// model computes what the network computes when it is not training
    /// </summary>
    internal sealed class ConvolutionalOnnxBuilder
    {
        private readonly ConvolutionalNetwork network;
        private readonly OnnxExportOptions options;
        private readonly OnnxGraph graph = new OnnxGraph();

        public ConvolutionalOnnxBuilder(ConvolutionalNetwork network, OnnxExportOptions options)
        {
            this.network = network;
            this.options = options;
        }

        public byte[] Build()
        {
            int inputSize = network.InputChannels * network.InputHeight * network.InputWidth;
            string value = options.InputName;
            bool flat = true;

            foreach (IConvolutionalLayer layer in network.Layers)
            {
                value = Emit(layer, value, ref flat);
            }

            if (network.OutputApplySoftMax)
            {
                value = graph.Node("Softmax", new[] { value }, OnnxGraph.IntAttribute("axis", 1));
            }

            graph.Node("Identity", new[] { value }, output: options.OutputName);
            return graph.ToModel(options.GraphName, options.InputName, inputSize, options.OutputName, network.OutputDimension,
                                 "Exported from an ANN.Net convolutional network");
        }

        /// <summary>
        /// Operators of one layer; <paramref name="flat"/> tells whether the value is [batch, values] or [batch, c, h, w]
        /// </summary>
        private string Emit(IConvolutionalLayer layer, string value, ref bool flat)
        {
            switch (layer)
            {
                case ConvolutionLayer convolution:
                    value = Spatial(value, ref flat);
                    string weights = graph.FloatTensor("conv_weights", Floats(convolution.Weights),
                                                       convolution.OutputShape.Channels, convolution.InputShape.Channels, convolution.KernelHeight, convolution.KernelWidth);
                    string[] inputs = convolution.Biases == null
                        ? new[] { value, weights }
                        : new[] { value, weights, graph.FloatTensor("conv_biases", Floats(convolution.Biases), convolution.OutputShape.Channels) };
                    value = graph.Node("Conv", inputs, w =>
                    {
                        OnnxGraph.IntsAttribute("kernel_shape", convolution.KernelHeight, convolution.KernelWidth)(w);
                        OnnxGraph.IntsAttribute("strides", convolution.Stride, convolution.Stride)(w);
                        OnnxGraph.IntsAttribute("pads", convolution.Padding, convolution.Padding, convolution.Padding, convolution.Padding)(w);
                    });
                    return OnnxActivations.Apply(graph, value, convolution.ActivationType);

                case PoolingLayer pooling:
                    value = Spatial(value, ref flat);
                    return graph.Node(pooling.Type == PoolingType.Max ? "MaxPool" : "AveragePool", new[] { value }, w =>
                    {
                        OnnxGraph.IntsAttribute("kernel_shape", pooling.Size, pooling.Size)(w);
                        OnnxGraph.IntsAttribute("strides", pooling.Stride, pooling.Stride)(w);
                        OnnxGraph.IntsAttribute("pads", pooling.Padding, pooling.Padding, pooling.Padding, pooling.Padding)(w);
                    });

                case BatchNormalizationLayer normalization:
                    value = Spatial(value, ref flat);
                    int channels = normalization.InputShape.Channels;
                    value = graph.Node("BatchNormalization", new[]
                    {
                        value,
                        graph.FloatTensor("bn_scale", Floats(normalization.Gamma), channels),
                        graph.FloatTensor("bn_bias", Floats(normalization.Beta), channels),
                        graph.FloatTensor("bn_mean", normalization.RunningMean.Select(x => (float)x).ToArray(), channels),
                        graph.FloatTensor("bn_variance", normalization.RunningVariance.Select(x => (float)x).ToArray(), channels)
                    }, OnnxGraph.FloatAttribute("epsilon", normalization.Epsilon));
                    return OnnxActivations.Apply(graph, value, normalization.ActivationType);

                case GlobalAveragePoolingLayer:
                    value = Spatial(value, ref flat);
                    return graph.Node("GlobalAveragePool", new[] { value });

                case ResidualBlockLayer residual:
                    value = Spatial(value, ref flat);
                    string main = value, shortcut = value;
                    foreach (IConvolutionalLayer inner in residual.Main) main = Emit(inner, main, ref flat);
                    foreach (IConvolutionalLayer inner in residual.Shortcut) shortcut = Emit(inner, shortcut, ref flat);
                    return OnnxActivations.Apply(graph, graph.Node("Add", new[] { main, shortcut }), residual.ActivationType);

                case DenseLayer dense:
                    if (!flat)
                    {
                        value = graph.Node("Flatten", new[] { value }, OnnxGraph.IntAttribute("axis", 1));
                        flat = true;
                    }

                    string denseWeights = graph.FloatTensor("dense_weights", Floats(dense.Weights), dense.InputShape.Size, dense.OutputShape.Size);
                    value = graph.Node("MatMul", new[] { value, denseWeights });
                    value = graph.Node("Add", new[] { value, graph.FloatTensor("dense_biases", Floats(dense.Biases), dense.OutputShape.Size) });
                    return OnnxActivations.Apply(graph, value, dense.ActivationType);

                default:
                    throw new NotSupportedException($"A {layer.GetType().Name} cannot be exported to ONNX");
            }
        }

        /// <summary>
        /// Turns the flat input into [batch, channels, height, width] the first time a spatial layer needs it
        /// </summary>
        private string Spatial(string value, ref bool flat)
        {
            if (!flat) return value;

            flat = false;
            string shape = graph.Int64Tensor("image_shape", new long[] { 0, network.InputChannels, network.InputHeight, network.InputWidth }, 4);
            return graph.Node("Reshape", new[] { value, shape });
        }

        private static float[] Floats(Parameters parameters) => parameters.Values.Select(x => (float)x).ToArray();
    }
}
