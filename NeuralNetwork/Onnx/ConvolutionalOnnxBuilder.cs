using System;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Convolutional;

namespace ANN.Net.Onnx
{
    /// <summary>
    /// ONNX graph of a convolutional network: the flat input is reshaped to [batch, channels, height, width], then
    /// Conv, MaxPool / AveragePool, Flatten and MatMul + Add with their activations follow, and Softmax for soft-max outputs
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
                switch (layer)
                {
                    case ConvolutionLayer convolution:
                        value = Spatial(value, ref flat);
                        string weights = graph.FloatTensor("conv_weights", Floats(convolution.Weights),
                                                           convolution.OutputShape.Channels, convolution.InputShape.Channels, convolution.KernelHeight, convolution.KernelWidth);
                        string biases = graph.FloatTensor("conv_biases", Floats(convolution.Biases), convolution.OutputShape.Channels);
                        value = graph.Node("Conv", new[] { value, weights, biases }, w =>
                        {
                            OnnxGraph.IntsAttribute("kernel_shape", convolution.KernelHeight, convolution.KernelWidth)(w);
                            OnnxGraph.IntsAttribute("strides", convolution.Stride, convolution.Stride)(w);
                            OnnxGraph.IntsAttribute("pads", convolution.Padding, convolution.Padding, convolution.Padding, convolution.Padding)(w);
                        });
                        value = OnnxActivations.Apply(graph, value, convolution.ActivationType);
                        break;

                    case PoolingLayer pooling:
                        value = Spatial(value, ref flat);
                        value = graph.Node(pooling.Type == PoolingType.Max ? "MaxPool" : "AveragePool", new[] { value }, w =>
                        {
                            OnnxGraph.IntsAttribute("kernel_shape", pooling.Size, pooling.Size)(w);
                            OnnxGraph.IntsAttribute("strides", pooling.Stride, pooling.Stride)(w);
                        });
                        break;

                    case DenseLayer dense:
                        if (!flat)
                        {
                            value = graph.Node("Flatten", new[] { value }, OnnxGraph.IntAttribute("axis", 1));
                            flat = true;
                        }

                        string denseWeights = graph.FloatTensor("dense_weights", Floats(dense.Weights), dense.InputShape.Size, dense.OutputShape.Size);
                        value = graph.Node("MatMul", new[] { value, denseWeights });
                        value = graph.Node("Add", new[] { value, graph.FloatTensor("dense_biases", Floats(dense.Biases), dense.OutputShape.Size) });
                        value = OnnxActivations.Apply(graph, value, dense.ActivationType);
                        break;

                    default:
                        throw new NotSupportedException($"A {layer.GetType().Name} cannot be exported to ONNX");
                }
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
