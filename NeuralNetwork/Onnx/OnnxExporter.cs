using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
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
    }

    /// <summary>
    /// Exports networks to ONNX (opset 13), so they can run in ONNX Runtime and other ONNX tools.
    /// The exported model computes the same values as <see cref="INetwork.Propagate"/>, including input
    /// normalization and the output soft-max of the network. Supports the networks of <c>NetworkFactory.BuildFFN</c>
    /// and <c>NetworkFactory.BuildGraph</c> without recurrent connections. Training settings (optimizers, learning
    /// rates, loss) are not part of ONNX; use <see cref="NetworkSerializer"/> to save a network for further training
    /// </summary>
    public static class OnnxExporter
    {
        public const string FileExtension = "onnx";

        private const float GELUActivationAlpha = ActivationFunctions.GELUActivation.DefaultAlpha;

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

            NetworkDocument document = NetworkSerializer.ToDocument(network);
            return new Builder(document, options).Build();
        }

        /// <summary>
        /// Evaluates the neurons depth by depth. All values computed so far live side by side in one
        /// [batch, columns] tensor; every group of neurons with the same depth and activation gathers the columns
        /// it reads, multiplies them by its weight matrix, adds the bias weights and appends its activations
        /// </summary>
        private sealed class Builder
        {
            private readonly NetworkDocument document;
            private readonly OnnxExportOptions options;
            private readonly OnnxGraph graph = new OnnxGraph();
            private readonly List<(int From, float Weight)>[] incoming;
            private readonly int[] column;
            private readonly int?[] depth;
            private readonly bool?[] fires;
            private string zeroColumn;

            public Builder(NetworkDocument document, OnnxExportOptions options)
            {
                this.document = document;
                this.options = options;

                int count = document.Neurons.Count;
                incoming = Enumerable.Range(0, count).Select(_ => new List<(int From, float Weight)>()).ToArray();
                column = Enumerable.Repeat(-1, count).ToArray();
                depth = new int?[count];
                fires = new bool?[count];

                foreach (SynapseDocument synapse in document.Synapses)
                {
                    if (synapse.Recurrent)
                        throw new NotSupportedException("Networks with recurrent connections cannot be exported to ONNX");

                    incoming[synapse.To].Add((synapse.From, synapse.Weight));
                }
            }

            private NodeType Type(int neuron) => document.Neurons[neuron].Type;

            public byte[] Build()
            {
                List<int> inputs = document.InputLayer.Where(x => Type(x) == NodeType.INPUT).ToList();
                if (inputs.Count == 0) throw new NotSupportedException("The network has no inputs");

                string values = Inputs(inputs);
                int width = inputs.Count;

                List<int> computed = Enumerable.Range(0, document.Neurons.Count)
                                               .Where(x => Type(x) == NodeType.HIDDEN || Type(x) == NodeType.OUTPUT)
                                               .Where(Fires)
                                               .ToList();

                foreach (var level in computed.GroupBy(Depth).OrderBy(x => x.Key))
                {
                    var parts = new List<string> { values };
                    foreach (var group in level.GroupBy(Activation))
                    {
                        List<int> neurons = group.ToList();
                        parts.Add(Activate(WeightedSum(values, neurons), group.Key));

                        foreach (int neuron in neurons)
                        {
                            column[neuron] = width++;
                        }
                    }

                    values = Concat(parts);
                }

                Outputs(values, width);

                return graph.ToModel(options.GraphName, options.InputName, inputs.Count, options.OutputName, document.OutputLayer.Count,
                                             $"Exported from an ANN.Net {document.Kind} network");
            }

            /// <summary>
            /// Input values, normalized and activated like the input neurons do
            /// </summary>
            private string Inputs(List<int> inputs)
            {
                string x = options.InputName;

                if (document.ShouldNormalize)
                {
                    if (document.MinMaxValues == null || document.MinMaxValues.Count < inputs.Count)
                        throw new NotSupportedException("The network normalizes its inputs but has no min/max value for every input");

                    float[] min = document.MinMaxValues.Take(inputs.Count).Select(v => v[0]).ToArray();
                    float[] range = document.MinMaxValues.Take(inputs.Count).Select(v => v[1] - v[0]).ToArray();
                    x = graph.Node("Sub", new[] { x, graph.FloatTensor("input_min", min, min.Length) });
                    x = graph.Node("Div", new[] { x, graph.FloatTensor("input_range", range, range.Length) });
                }

                ActivationTypes[] activations = inputs.Select(i => document.Neurons[i].Activation).ToArray();
                if (activations.All(a => a == ActivationTypes.Identity))
                {
                    for (int i = 0; i < inputs.Count; i++) column[inputs[i]] = i;
                    return x;
                }

                var parts = new List<string>();
                int width = 0;
                foreach (var group in Enumerable.Range(0, inputs.Count).GroupBy(i => activations[i]))
                {
                    long[] indexes = group.Select(i => (long)i).ToArray();
                    parts.Add(Activate(Gather(x, indexes), group.Key));
                    foreach (int i in group) column[inputs[i]] = width++;
                }

                return Concat(parts);
            }

            private void Outputs(string values, int width)
            {
                // an output without a complete set of inputs never fires in ANN.Net and keeps the value 0
                long zero = -1;
                if (document.OutputLayer.Exists(x => column[x] < 0))
                {
                    values = Concat(new List<string> { values, ZeroColumn() });
                    zero = width;
                }

                long[] columns = document.OutputLayer.Select(x => column[x] < 0 ? zero : column[x]).ToArray();
                string output = Gather(values, columns);

                if (document.OutputApplySoftMax)
                {
                    // ANN.Net divides every output by the sum of all outputs
                    float[] ones = Enumerable.Repeat(1f, columns.Length).ToArray();
                    string sum = graph.Node("MatMul", new[] { output, graph.FloatTensor("ones", ones, ones.Length, 1) });
                    output = graph.Node("Div", new[] { output, sum });
                }

                graph.Node("Identity", new[] { output }, output: options.OutputName);
            }

            private string WeightedSum(string values, List<int> neurons)
            {
                List<int> sources = neurons.SelectMany(n => incoming[n])
                                           .Where(s => Type(s.From) != NodeType.BIAS)
                                           .Select(s => column[s.From])
                                           .Distinct()
                                           .OrderBy(c => c)
                                           .ToList();
                var row = sources.Select((c, i) => (c, i)).ToDictionary(x => x.c, x => x.i);

                var weights = new float[sources.Count * neurons.Count];
                var bias = new float[neurons.Count];

                for (int n = 0; n < neurons.Count; n++)
                {
                    foreach (var (from, weight) in incoming[neurons[n]])
                    {
                        if (Type(from) == NodeType.BIAS) bias[n] += weight;
                        else weights[(row[column[from]] * neurons.Count) + n] += weight;
                    }
                }

                string sum;
                if (sources.Count > 0)
                {
                    string selected = Gather(values, sources.Select(c => (long)c).ToArray());
                    sum = graph.Node("MatMul", new[] { selected, graph.FloatTensor("weights", weights, sources.Count, neurons.Count) });
                }
                else
                {
                    sum = ZeroColumn(); // fed only by bias neurons; broadcasts to [batch, neurons] below
                }

                if (sources.Count == 0 || bias.Any(b => b != 0))
                {
                    sum = graph.Node("Add", new[] { sum, graph.FloatTensor("bias", bias, bias.Length) });
                }

                return sum;
            }

            private string Activate(string value, ActivationTypes activation)
            {
                switch (activation)
                {
                    case ActivationTypes.Identity:
                        return value;
                    case ActivationTypes.Sigmoid:
                        return graph.Node("Sigmoid", new[] { value });
                    case ActivationTypes.HyperbolicTangens:
                        return graph.Node("Tanh", new[] { value });
                    case ActivationTypes.ArcusTangens:
                        return graph.Node("Atan", new[] { value });
                    case ActivationTypes.RELU:
                        return graph.Node("Relu", new[] { value });
                    case ActivationTypes.LeakyRELU:
                        return graph.Node("LeakyRelu", new[] { value }, OnnxGraph.FloatAttribute("alpha", 0.01f));
                    case ActivationTypes.BinaryRELU:
                        string positive = graph.Node("GreaterOrEqual", new[] { value, Scalar(0f) });
                        return graph.Node("Cast", new[] { positive }, OnnxGraph.IntAttribute("to", 1));
                    case ActivationTypes.Sinusoid:
                        return graph.Node("Sin", new[] { value });
                    case ActivationTypes.SoftSign:
                        return graph.Node("Softsign", new[] { value });
                    case ActivationTypes.Gaussian:
                        // e^(-x²)
                        return graph.Node("Exp", new[] { graph.Node("Neg", new[] { Square(value) }) });
                    case ActivationTypes.SWISH:
                        // x * sigmoid(x)
                        return graph.Node("Mul", new[] { value, graph.Node("Sigmoid", new[] { value }) });
                    case ActivationTypes.ISRU:
                        // x / sqrt(1 + x²), alpha = 1
                        return graph.Node("Div", new[] { value, graph.Node("Sqrt", new[] { graph.Node("Add", new[] { Square(value), Scalar(1f) }) }) });
                    case ActivationTypes.SQNL:
                        // with c = clip(x, -2, 2): c - c * |c| / 4 covers all four pieces
                        string clipped = graph.Node("Clip", new[] { value, Scalar(-2f), Scalar(2f) });
                        string quarter = graph.Node("Mul", new[] { graph.Node("Mul", new[] { clipped, graph.Node("Abs", new[] { clipped }) }), Scalar(0.25f) });
                        return graph.Node("Sub", new[] { clipped, quarter });
                    case ActivationTypes.GELU:
                        // tanh approximation used by ANN.Net: 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + alpha * x³)))
                        string cube = graph.Node("Mul", new[] { Square(value), value });
                        string inner = graph.Node("Add", new[] { value, graph.Node("Mul", new[] { cube, Scalar(GELUActivationAlpha) }) });
                        string tanh = graph.Node("Tanh", new[] { graph.Node("Mul", new[] { inner, Scalar((float)Math.Sqrt(2 / Math.PI)) }) });
                        string half = graph.Node("Mul", new[] { value, Scalar(0.5f) });
                        return graph.Node("Mul", new[] { half, graph.Node("Add", new[] { tanh, Scalar(1f) }) });
                    default:
                        throw new NotSupportedException($"The {activation} activation cannot be exported to ONNX");
                }
            }

            private string Scalar(float value)
            {
                return graph.FloatTensor("constant", new[] { value });
            }

            private string Square(string value)
            {
                return graph.Node("Mul", new[] { value, value });
            }

            /// <summary>
            /// Activation actually applied by the neuron: outputs of a soft-max network are not activated
            /// </summary>
            private ActivationTypes Activation(int neuron)
            {
                NeuronDocument document = this.document.Neurons[neuron];
                return document.ApplyActivation == false ? ActivationTypes.Identity : document.Activation;
            }

            /// <summary>
            /// A neuron fires once every input synapse delivered, so it needs at least one input and inputs that all fire
            /// </summary>
            private bool Fires(int neuron)
            {
                if (fires[neuron] is bool known) return known;

                NodeType type = Type(neuron);
                bool result = type == NodeType.INPUT || type == NodeType.BIAS
                    || (incoming[neuron].Count > 0 && incoming[neuron].All(s => Fires(s.From)));

                fires[neuron] = result;
                return result;
            }

            /// <summary>
            /// Inputs have depth 0, every other neuron is one deeper than its deepest non-bias input
            /// </summary>
            private int Depth(int neuron)
            {
                if (depth[neuron] is int known) return known;

                int result = Type(neuron) == NodeType.INPUT ? 0
                    : 1 + incoming[neuron].Where(s => Type(s.From) != NodeType.BIAS).Select(s => Depth(s.From)).DefaultIfEmpty(0).Max();

                depth[neuron] = result;
                return result;
            }

            private string Gather(string values, long[] columns)
            {
                return graph.Node("Gather", new[] { values, graph.Int64Tensor("columns", columns, columns.Length) }, OnnxGraph.IntAttribute("axis", 1));
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
