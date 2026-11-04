using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Compact;
using ANN.Net.Onnx;
using ANN.Net.Serialization;

namespace ANN.Net
{
    /// <summary>
    /// Names used in an exported ONNX model
    /// </summary>
    public sealed class OnnxExportOptions
    {
        /// <summary>
        /// Name of the float input tensor of shape [batch, inputs]
        /// </summary>
        public string InputName { get; set; } = "input";

        /// <summary>
        /// Name of the float output tensor of shape [batch, outputs]
        /// </summary>
        public string OutputName { get; set; } = "output";

        public string GraphName { get; set; } = "ann_net";

        /// <summary>
        /// Number format of the weight matrices in the file: float16 weights are cast to float, int8 weights are
        /// dequantized with one scale per neuron or filter. Graph and layered networks then compute what the compact
        /// network of the same precision computes; convolutional networks quantize the weights before batch
        /// normalization, which compact networks fold into them first, so the two differ by a little rounding
        /// </summary>
        public WeightPrecision WeightPrecision { get; set; } = WeightPrecision.Single;
    }

    /// <summary>
    /// Exports networks to ONNX (opset 13), so they can run in ONNX Runtime and other ONNX tools.
    /// The exported model computes the same values as <see cref="INetwork.Propagate"/>, including input
    /// normalization and the output soft-max of the network. Supports the networks of <c>NetworkFactory.BuildFFN</c>
    /// and <c>NetworkFactory.BuildGraph</c> without recurrent connections, and of <c>NetworkFactory.BuildConvolutional</c>. Training settings (optimizers, learning
    /// rates, loss) are not part of ONNX; use <see cref="NetworkSerializer"/> to save a network for further training
    /// </summary>
    public static class OnnxExporter
    {
        public const string FileExtension = "onnx";


        public static void Export(INetwork network, string filePath, OnnxExportOptions options = null)
        {
            ArgumentException.ThrowIfNullOrEmpty(filePath);

            byte[] model = Export(network, options);
            File.WriteAllBytes(filePath, model);
        }

        public static void Export(INetwork network, Stream stream, OnnxExportOptions options = null)
        {
            ArgumentNullException.ThrowIfNull(stream);

            stream.Write(Export(network, options));
        }

        /// <summary>
        /// Encodes the network as an ONNX model
        /// </summary>
        public static byte[] Export(INetwork network, OnnxExportOptions options = null)
        {
            ArgumentNullException.ThrowIfNull(network);
            options ??= new OnnxExportOptions();

            if (network is Convolutional.ConvolutionalNetwork convolutional)
                return new ConvolutionalOnnxBuilder(convolutional, options).Build();

            NetworkDocument document = NetworkSerializer.ToDocument(network);
            return new Builder(document, options).Build();
        }

        /// <summary>
        /// ONNX operators of a <see cref="GraphPlan"/>: every level gathers the columns of its groups, multiplies them by
        /// the group weights, adds the biases, activates and appends the results to the [batch, columns] tensor
        /// </summary>
        private sealed class Builder
        {
            private readonly GraphPlan plan;
            private readonly NetworkDocument document;
            private readonly OnnxExportOptions options;
            private readonly OnnxGraph graph = new OnnxGraph();
            private string zeroColumn;

            public Builder(NetworkDocument document, OnnxExportOptions options)
            {
                this.document = document;
                this.options = options;
                plan = GraphPlan.Create(document);
            }

            public byte[] Build()
            {
                string values = Inputs();

                foreach (IReadOnlyList<GraphPlan.NeuronGroup> level in plan.Levels)
                {
                    var parts = new List<string> { values };
                    foreach (GraphPlan.NeuronGroup group in level)
                    {
                        parts.Add(OnnxActivations.Apply(graph, WeightedSum(values, group), group.Activation));
                    }

                    values = Concat(parts);
                }

                Outputs(values);

                return graph.ToModel(options.GraphName, options.InputName, plan.InputCount, options.OutputName, plan.OutputColumns.Length,
                                     $"Exported from an ANN.Net {document.Kind} network");
            }

            /// <summary>
            /// Input values, normalized and activated like the input neurons do
            /// </summary>
            private string Inputs()
            {
                string x = options.InputName;

                if (plan.InputMinimum != null)
                {
                    x = graph.Node("Sub", new[] { x, graph.FloatTensor("input_min", plan.InputMinimum, plan.InputCount) });
                    x = graph.Node("Div", new[] { x, graph.FloatTensor("input_range", plan.InputRange, plan.InputCount) });
                }

                if (plan.InputGroups.Count == 0) return x;

                return Concat(plan.InputGroups.Select(g => OnnxActivations.Apply(graph, Gather(x, g.Positions), g.Activation)).ToList());
            }

            private void Outputs(string values)
            {
                int[] columns = plan.OutputColumns;
                if (columns.Contains(-1))
                {
                    values = Concat(new List<string> { values, ZeroColumn() });
                    columns = columns.Select(x => x < 0 ? plan.Width : x).ToArray();
                }

                string output = Gather(values, columns);

                if (plan.DivideBySum)
                {
                    // ANN.Net divides every output by the sum of all outputs
                    float[] ones = Enumerable.Repeat(1f, columns.Length).ToArray();
                    string sum = graph.Node("MatMul", new[] { output, graph.FloatTensor("ones", ones, ones.Length, 1) });
                    output = graph.Node("Div", new[] { output, sum });
                }

                graph.Node("Identity", new[] { output }, output: options.OutputName);
            }

            private string WeightedSum(string values, GraphPlan.NeuronGroup group)
            {
                string sum = group.Sources.Length > 0
                    ? graph.Node("MatMul", new[] { Gather(values, group.Sources), graph.WeightTensor("weights", group.Weights, options.WeightPrecision, 1, group.Sources.Length, group.Count) })
                    : ZeroColumn(); // fed only by bias neurons; broadcasts to [batch, neurons] below

                if (group.Sources.Length == 0 || group.Bias.Any(b => b != 0))
                {
                    sum = graph.Node("Add", new[] { sum, graph.FloatTensor("bias", group.Bias, group.Count) });
                }

                return sum;
            }

            private string Gather(string values, int[] columns)
            {
                return graph.Node("Gather", new[] { values, graph.Int64Tensor("columns", columns.Select(x => (long)x).ToArray(), columns.Length) }, OnnxGraph.IntAttribute("axis", 1));
            }

            private string Concat(List<string> parts)
            {
                return parts.Count == 1 ? parts[0] : graph.Node("Concat", parts.ToArray(), OnnxGraph.IntAttribute("axis", 1));
            }

            /// <summary>
            /// [batch, 1] tensor of zeros
            /// </summary>
            private string ZeroColumn()
            {
                if (zeroColumn != null) return zeroColumn;

                string shape = graph.Node("Shape", new[] { options.InputName });
                string batch = graph.Node("Gather", new[] { shape, graph.Int64Tensor("batch_index", new[] { 0L }, 1) }, OnnxGraph.IntAttribute("axis", 0));
                string columnShape = graph.Node("Concat", new[] { batch, graph.Int64Tensor("one", new[] { 1L }, 1) }, OnnxGraph.IntAttribute("axis", 0));
                zeroColumn = graph.Node("ConstantOfShape", new[] { columnShape }, OnnxGraph.FloatTensorAttribute("value", 0f));
                return zeroColumn;
            }
        }
    }
}
