using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Serialization;

namespace ANN.Net.Compact
{
    /// <summary>
    /// A neuron-graph network as a few matrix operations. All values computed so far live side by side in one row of
    /// columns: the inputs first, then level after level of neurons. A level holds the neurons of one depth; within it
    /// every group of neurons with the same activation reads the columns it needs, multiplies them by its weight matrix
    /// and adds its bias weights. Used by the ONNX export and by compact networks
    /// </summary>
    internal sealed class GraphPlan
    {
        private GraphPlan()
        {
        }

        public int InputCount { get; private set; }

        /// <summary>
        /// Minimum and range of every input when the network normalizes its inputs, otherwise null
        /// </summary>
        public float[] InputMinimum { get; private set; }

        public float[] InputRange { get; private set; }

        /// <summary>
        /// Groups of input positions with the same activation, in column order; empty when every input is the identity
        /// and the inputs are the first columns as they are
        /// </summary>
        public IReadOnlyList<InputGroup> InputGroups { get; private set; }

        public IReadOnlyList<IReadOnlyList<NeuronGroup>> Levels { get; private set; }

        /// <summary>
        /// Number of columns after the last level
        /// </summary>
        public int Width { get; private set; }

        /// <summary>
        /// Column of every output, or -1 for an output that never fires and stays 0
        /// </summary>
        public int[] OutputColumns { get; private set; }

        /// <summary>
        /// The network divides every output by the sum of all outputs (its "soft-max")
        /// </summary>
        public bool DivideBySum { get; private set; }

        internal sealed record InputGroup(ActivationTypes Activation, int[] Positions);

        /// <param name="Sources">Columns the group reads</param>
        /// <param name="Weights">Sources x neurons, row by row</param>
        /// <param name="Bias">Sum of the bias neuron weights of every neuron</param>
        /// <param name="FirstColumn">Column of the first neuron; the others follow</param>
        internal sealed record NeuronGroup(int[] Sources, float[] Weights, float[] Bias, ActivationTypes Activation, int FirstColumn)
        {
            public int Count => Bias.Length;
        }

        /// <summary>
        /// The same inputs and outputs with other levels
        /// </summary>
        internal GraphPlan With(IReadOnlyList<IReadOnlyList<NeuronGroup>> levels, int width, int[] outputColumns) => new GraphPlan
        {
            InputCount = InputCount,
            InputMinimum = InputMinimum,
            InputRange = InputRange,
            InputGroups = InputGroups,
            Levels = levels,
            Width = width,
            OutputColumns = outputColumns,
            DivideBySum = DivideBySum
        };

        public static GraphPlan Create(NetworkDocument document)
        {
            return new Planner(document).Plan();
        }

        private sealed class Planner
        {
            private readonly NetworkDocument document;
            private readonly List<(int From, float Weight)>[] incoming;
            private readonly int[] column;
            private readonly int?[] depth;
            private readonly bool?[] fires;

            public Planner(NetworkDocument document)
            {
                this.document = document;

                int count = document.Neurons.Count;
                incoming = Enumerable.Range(0, count).Select(_ => new List<(int From, float Weight)>()).ToArray();
                column = Enumerable.Repeat(-1, count).ToArray();
                depth = new int?[count];
                fires = new bool?[count];

                foreach (SynapseDocument synapse in document.Synapses)
                {
                    if (synapse.Recurrent)
                        throw new NotSupportedException("Networks with recurrent connections are not supported");

                    incoming[synapse.To].Add((synapse.From, synapse.Weight));
                }
            }

            private NodeType Type(int neuron) => document.Neurons[neuron].Type;

            public GraphPlan Plan()
            {
                List<int> inputs = document.InputLayer.Where(x => Type(x) == NodeType.INPUT).ToList();
                if (inputs.Count == 0) throw new NotSupportedException("The network has no inputs");

                var plan = new GraphPlan { InputCount = inputs.Count, DivideBySum = document.OutputApplySoftMax };

                if (document.ShouldNormalize)
                {
                    if (document.MinMaxValues == null || document.MinMaxValues.Count < inputs.Count)
                        throw new NotSupportedException("The network normalizes its inputs but has no min/max value for every input");

                    plan.InputMinimum = document.MinMaxValues.Take(inputs.Count).Select(v => v[0]).ToArray();
                    plan.InputRange = document.MinMaxValues.Take(inputs.Count).Select(v => v[1] - v[0]).ToArray();
                }

                plan.InputGroups = PlanInputs(inputs);
                int width = inputs.Count;

                List<int> computed = Enumerable.Range(0, document.Neurons.Count)
                                               .Where(x => Type(x) == NodeType.HIDDEN || Type(x) == NodeType.OUTPUT)
                                               .Where(Fires)
                                               .ToList();

                var levels = new List<IReadOnlyList<NeuronGroup>>();
                foreach (var level in computed.GroupBy(Depth).OrderBy(x => x.Key))
                {
                    var groups = new List<NeuronGroup>();
                    foreach (var group in level.GroupBy(Activation))
                    {
                        List<int> neurons = group.ToList();
                        groups.Add(PlanGroup(neurons, group.Key, width));

                        foreach (int neuron in neurons)
                        {
                            column[neuron] = width++;
                        }
                    }

                    levels.Add(groups);
                }

                plan.Levels = levels;
                plan.Width = width;

                // an output without a complete set of inputs never fires in ANN.Net and keeps the value 0
                plan.OutputColumns = document.OutputLayer.Select(x => column[x]).ToArray();
                return plan;
            }

            private List<InputGroup> PlanInputs(List<int> inputs)
            {
                ActivationTypes[] activations = inputs.Select(i => document.Neurons[i].Activation).ToArray();
                if (activations.All(a => a == ActivationTypes.Identity))
                {
                    for (int i = 0; i < inputs.Count; i++) column[inputs[i]] = i;
                    return new List<InputGroup>();
                }

                var groups = new List<InputGroup>();
                int width = 0;
                foreach (var group in Enumerable.Range(0, inputs.Count).GroupBy(i => activations[i]))
                {
                    groups.Add(new InputGroup(group.Key, group.ToArray()));
                    foreach (int i in group) column[inputs[i]] = width++;
                }

                return groups;
            }

            private NeuronGroup PlanGroup(List<int> neurons, ActivationTypes activation, int firstColumn)
            {
                int[] sources = neurons.SelectMany(n => incoming[n])
                                       .Where(s => Type(s.From) != NodeType.BIAS)
                                       .Select(s => column[s.From])
                                       .Distinct()
                                       .OrderBy(c => c)
                                       .ToArray();
                var row = sources.Select((c, i) => (c, i)).ToDictionary(x => x.c, x => x.i);

                var weights = new float[sources.Length * neurons.Count];
                var bias = new float[neurons.Count];

                for (int n = 0; n < neurons.Count; n++)
                {
                    foreach (var (from, weight) in incoming[neurons[n]])
                    {
                        if (Type(from) == NodeType.BIAS) bias[n] += weight;
                        else weights[(row[column[from]] * neurons.Count) + n] += weight;
                    }
                }

                return new NeuronGroup(sources, weights, bias, activation, firstColumn);
            }

            /// <summary>
            /// Activation actually applied by the neuron: outputs of a soft-max network are not activated
            /// </summary>
            private ActivationTypes Activation(int neuron)
            {
                NeuronDocument neuronDocument = document.Neurons[neuron];
                return neuronDocument.ApplyActivation == false ? ActivationTypes.Identity : neuronDocument.Activation;
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
        }
    }
}
