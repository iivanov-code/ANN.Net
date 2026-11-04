using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Models;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Utils;

namespace ANN.Net.Nested
{
    /// <summary>
    /// Network of networks: every node holds a vector network, receives the output vectors of its input nodes
    /// (concatenated or summed) and passes its own output vector on. Nodes run in dependency order; backpropagation
    /// visits them in reverse, adds up the gradients of every node's consumers and hands them to the network inside,
    /// which trains itself and returns the gradient of its inputs
    /// </summary>
    internal sealed class NestedNetwork : ANNetwork, INestedNetwork
    {
        private readonly Node[] order;
        private readonly Dictionary<uint, Node> nodes;
        private readonly uint[] outputs;
        private readonly ILossFunction lossFunction;
        private readonly List<LearningRateOptimizer> optimizers;
        private Quad[] input;

        private NestedNetwork(int inputSize, Node[] order, uint[] outputs, LossFunctionTypes lossType)
        {
            InputSize = inputSize;
            this.order = order;
            nodes = order.ToDictionary(x => x.Id);
            this.outputs = outputs;
            LossType = lossType;
            lossFunction = NetworkUtils.GetLossFunction(lossType);
            optimizers = order.SelectMany(x => x.Network.Optimizers).ToList();
            OutputSize = outputs.Sum(SizeOf);
            Errors = new Quad[OutputSize];
        }

        internal LossFunctionTypes LossType { get; }

        internal IEnumerable<(uint Id, IVectorNetwork Network, uint[] Inputs, NodeCombination Combination)> NodeList =>
            order.Select(x => (x.Id, x.Network, x.Inputs, x.Combination));

        internal IReadOnlyList<uint> OutputIds => outputs;

        public int InputSize { get; }

        public int OutputSize { get; }

        public long ParameterCount => order.Sum(x => x.Network.ParameterCount);

        public ICollection<LearningRateOptimizer> Optimizers => optimizers;

        public IReadOnlyDictionary<uint, IVectorNetwork> Nodes => order.ToDictionary(x => x.Id, x => x.Network);

        public IReadOnlyDictionary<uint, Quad[]> NodeValues => order.Where(x => x.Value != null).ToDictionary(x => x.Id, x => x.Value);

        public Quad[] Values { get; private set; } = Array.Empty<Quad>();

        public Quad[] Errors { get; }

        public Quad[] InputGradients { get; private set; } = Array.Empty<Quad>();

        public IReadOnlyList<Tuple<Quad, Quad>> MinMaxValues => null;

        private int SizeOf(uint id) => id == NestedNetworkSettings.InputNode ? InputSize : nodes[id].Network.OutputSize;

        private Quad[] ValueOf(uint id) => id == NestedNetworkSettings.InputNode ? input : nodes[id].Value;

        public static NestedNetwork Build(NestedNetworkSettings settings)
        {
            ArgumentNullException.ThrowIfNull(settings);
            if (settings.InputSize < 1) throw new ArgumentException("The input size must be at least 1", nameof(settings));
            if (settings.Nodes == null || settings.Outputs == null || settings.Outputs.Count == 0)
                throw new ArgumentException("Nodes and at least one output are required", nameof(settings));

            var byId = new Dictionary<uint, NestedNode>();
            var networks = new HashSet<IVectorNetwork>(ReferenceEqualityComparer.Instance);
            foreach (NestedNode node in settings.Nodes)
            {
                if (node == null) throw new ArgumentException("Nodes must not contain null", nameof(settings));
                if (node.Id == NestedNetworkSettings.InputNode)
                    throw new ArgumentException($"Node id {NestedNetworkSettings.InputNode} is the input of the network", nameof(settings));
                if (!byId.TryAdd(node.Id, node)) throw new ArgumentException($"Duplicate node id {node.Id}", nameof(settings));
                if (node.Network == null) throw new ArgumentException($"Node {node.Id} has no network", nameof(settings));
                if (!networks.Add(node.Network))
                    throw new ArgumentException($"The network of node {node.Id} is inside another node too; every node needs its own network", nameof(settings));
                if (node.Inputs == null || node.Inputs.Count == 0) throw new ArgumentException($"Node {node.Id} has no inputs", nameof(settings));
                if (node.Inputs.Any(x => x != NestedNetworkSettings.InputNode && !byId.ContainsKey(x) && !settings.Nodes.Any(n => n?.Id == x)))
                    throw new ArgumentException($"Node {node.Id} reads an unknown node", nameof(settings));
            }

            foreach (uint output in settings.Outputs)
            {
                if (output != NestedNetworkSettings.InputNode && !byId.ContainsKey(output))
                    throw new ArgumentException($"Unknown output node {output}", nameof(settings));
            }

            // only nodes that lead to an output are computed
            var needed = new HashSet<uint>();
            var stack = new Stack<uint>(settings.Outputs.Where(x => x != NestedNetworkSettings.InputNode));
            while (stack.Count > 0)
            {
                uint id = stack.Pop();
                if (!needed.Add(id)) continue;
                foreach (uint source in byId[id].Inputs.Where(x => x != NestedNetworkSettings.InputNode)) stack.Push(source);
            }

            Node[] order = Sort(settings.Nodes.Where(x => needed.Contains(x.Id)).ToList());
            var network = new NestedNetwork(settings.InputSize, order, settings.Outputs.ToArray(), settings.LossType);
            foreach (Node node in order) network.CheckSizes(node);
            return network;
        }

        /// <summary>
        /// Dependency order, keeping the order of the settings where it is free
        /// </summary>
        private static Node[] Sort(List<NestedNode> nodes)
        {
            var remaining = nodes.ToDictionary(x => x.Id, x => x.Inputs.Where(s => s != NestedNetworkSettings.InputNode).Distinct().Count());
            var consumers = nodes.SelectMany(n => n.Inputs.Where(s => s != NestedNetworkSettings.InputNode).Distinct().Select(s => (Source: s, Consumer: n.Id)))
                                 .ToLookup(x => x.Source, x => x.Consumer);
            var ready = new SortedSet<int>(nodes.Select((n, i) => (n, i)).Where(x => remaining[x.n.Id] == 0).Select(x => x.i));
            var position = nodes.Select((n, i) => (n.Id, i)).ToDictionary(x => x.Id, x => x.i);
            var result = new List<Node>();

            while (ready.Count > 0)
            {
                int index = ready.Min;
                ready.Remove(index);
                NestedNode node = nodes[index];
                result.Add(new Node(node.Id, node.Network, node.Inputs.ToArray(), node.Combination));

                foreach (uint consumer in consumers[node.Id])
                {
                    if (--remaining[consumer] == 0) ready.Add(position[consumer]);
                }
            }

            if (result.Count != nodes.Count)
                throw new ArgumentException("The nodes form a cycle; a nested network must be acyclic");

            return result.ToArray();
        }

        private void CheckSizes(Node node)
        {
            int[] sizes = node.Inputs.Select(SizeOf).ToArray();
            int expected = node.Network.InputSize;
            bool fits = node.Combination == NodeCombination.Sum ? sizes.All(x => x == expected) : sizes.Sum() == expected;
            if (!fits)
            {
                throw new ArgumentException(
                    $"Node {node.Id} needs {expected} values but its inputs give {string.Join(node.Combination == NodeCombination.Sum ? ", " : " + ", sizes)} ({node.Combination})");
            }
        }

        public Quad[] Propagate(params Quad[] values)
        {
            ArgumentNullException.ThrowIfNull(values);
            if (values.Length != InputSize)
                throw new ArgumentException($"Expected {InputSize} input values but found {values.Length}", nameof(values));

            input = (Quad[])values.Clone();
            foreach (Node node in order)
            {
                node.Value = (Quad[])node.Network.Propagate(Gather(node)).Clone();
            }

            Values = outputs.SelectMany(ValueOf).ToArray();
            return (Quad[])Values.Clone();
        }

        private Quad[] Gather(Node node)
        {
            if (node.Combination == NodeCombination.Concatenate) return node.Inputs.SelectMany(ValueOf).ToArray();

            var sum = new Quad[node.Network.InputSize];
            foreach (uint source in node.Inputs)
            {
                Quad[] value = ValueOf(source);
                for (int i = 0; i < sum.Length; i++) sum[i] += value[i];
            }

            return sum;
        }

        public Quad Backpropagate(params Quad[] targets)
        {
            GradientUtils.CheckLength(targets, OutputSize, nameof(targets));
            if (input == null) throw new InvalidOperationException("Propagate must be called before Backpropagate");

            var errors = new Quad[OutputSize];
            var results = new List<ErrorResult>(OutputSize);
            for (int i = 0; i < errors.Length; i++)
            {
                errors[i] = Values[i] - targets[i];
                results.Add(new ErrorResult { Target = targets[i], Predicted = Values[i] });
            }

            BackpropagateGradient(errors);
            return lossFunction.CalculateLoss(results);
        }

        public Quad[] BackpropagateGradient(Quad[] outputGradients)
        {
            GradientUtils.CheckLength(outputGradients, OutputSize, nameof(outputGradients));
            if (input == null) throw new InvalidOperationException("Propagate must be called before BackpropagateGradient");

            outputGradients.CopyTo(Errors, 0);
            var inputGradients = new Quad[InputSize];
            foreach (Node node in order) node.Gradient = new Quad[node.Network.OutputSize];

            int offset = 0;
            foreach (uint id in outputs)
            {
                Add(id, outputGradients.AsSpan(offset, SizeOf(id)), inputGradients);
                offset += SizeOf(id);
            }

            for (int n = order.Length - 1; n >= 0; n--)
            {
                Node node = order[n];
                Quad[] gradients = node.Network.BackpropagateGradient(node.Gradient);

                if (node.Combination == NodeCombination.Sum)
                {
                    foreach (uint source in node.Inputs) Add(source, gradients, inputGradients);
                    continue;
                }

                int start = 0;
                foreach (uint source in node.Inputs)
                {
                    Add(source, gradients.AsSpan(start, SizeOf(source)), inputGradients);
                    start += SizeOf(source);
                }
            }

            InputGradients = inputGradients;
            return (Quad[])inputGradients.Clone();
        }

        private void Add(uint id, ReadOnlySpan<Quad> gradients, Quad[] inputGradients)
        {
            Quad[] target = id == NestedNetworkSettings.InputNode ? inputGradients : nodes[id].Gradient;
            for (int i = 0; i < gradients.Length; i++) target[i] += gradients[i];
        }

        private sealed class Node
        {
            public Node(uint id, IVectorNetwork network, uint[] inputs, NodeCombination combination)
            {
                Id = id;
                Network = network;
                Inputs = inputs;
                Combination = combination;
            }

            public uint Id { get; }

            public IVectorNetwork Network { get; }

            public uint[] Inputs { get; }

            public NodeCombination Combination { get; }

            public Quad[] Value { get; set; }

            public Quad[] Gradient { get; set; }
        }
    }
}
