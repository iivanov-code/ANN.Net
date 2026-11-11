using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Collections;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Connections;
using ANN.Net.IdentityGenerators;
using ANN.Net.Neurons;
using ANN.Net.Serialization;
using ANN.Net.Utils;

namespace ANN.Net
{
    /// <summary>
    /// Feed-forward network with an arbitrary acyclic topology. A neuron fires once every one of its input
    /// synapses has delivered, so any directed acyclic graph propagates and backpropagates correctly as long as
    /// every neuron lies on a path from an input to an output; the builder prunes everything else
    /// </summary>
    [Serializable]
    internal sealed class GraphNetwork : ANNetwork, IGraphNetwork
    {
        private readonly Network network;
        private readonly Dictionary<uint, ActiveSynapse> synapses;
        private readonly List<uint> hiddenNodeIds;

        private GraphNetwork(Network network, Dictionary<uint, ActiveSynapse> synapses, List<uint> hiddenNodeIds)
        {
            this.network = network;
            this.synapses = synapses;
            this.hiddenNodeIds = hiddenNodeIds;
        }

        internal Network Network => network;

        internal IReadOnlyDictionary<uint, ActiveSynapse> Synapses => synapses;

        /// <summary>
        /// Wraps a network that was rebuilt from a file
        /// </summary>
        internal static GraphNetwork FromNetwork(Network network, Dictionary<uint, ActiveSynapse> synapses, List<uint> hiddenNodeIds)
        {
            return new GraphNetwork(network, synapses, hiddenNodeIds);
        }

        public Quad[] Values => network.Values;

        public Quad[] Errors => network.Errors;

        public Quad[] InputGradients => network.InputGradients;

        public IReadOnlyList<Tuple<Quad, Quad>> MinMaxValues => network.MinMaxValues;

        public ICollection<LearningRateOptimizer> Optimizers => network.Optimizers;

        public IReadOnlyDictionary<uint, Quad> ConnectionWeights => synapses.ToDictionary(x => x.Key, x => x.Value.Weight);

        public IReadOnlyCollection<uint> HiddenNodeIds => hiddenNodeIds;

        public Quad[] Propagate(params Quad[] values) => network.Propagate(values);

        public Quad Backpropagate(params Quad[] targets) => network.Backpropagate(targets);

        public int InputSize => network.InputSize;

        public int OutputSize => network.OutputSize;

        public long ParameterCount => synapses.Count;

        public Quad[] BackpropagateGradient(Quad[] outputGradients) => network.BackpropagateGradient(outputGradients);

        /// <summary>
        /// Nodes and connections of a valid graph after pruning, in the order of the settings
        /// </summary>
        private sealed record Topology(List<GraphNode> Inputs, List<GraphNode> Biases, List<GraphNode> Hidden, List<GraphNode> Outputs, List<GraphConnection> Connections);

        private static Topology Prepare(GraphNetworkSettings settings)
        {
            if (settings == null) throw new ArgumentNullException(nameof(settings));
            if (settings.Nodes == null) throw new ArgumentException("Nodes are required", nameof(settings));

            var nodes = new Dictionary<uint, GraphNode>();
            foreach (GraphNode node in settings.Nodes)
            {
                if (node == null) throw new ArgumentException("Nodes must not contain null", nameof(settings));
                if (!nodes.TryAdd(node.Id, node))
                    throw new ArgumentException($"Duplicate node id {node.Id}", nameof(settings));
            }

            List<GraphNode> inputs = settings.Nodes.Where(x => x.Type == NodeType.INPUT).ToList();
            List<GraphNode> biases = settings.Nodes.Where(x => x.Type == NodeType.BIAS).ToList();
            List<GraphNode> outputs = settings.Nodes.Where(x => x.Type == NodeType.OUTPUT).ToList();

            if (inputs.Count == 0) throw new ArgumentException("At least one input node is required", nameof(settings));
            if (outputs.Count == 0) throw new ArgumentException("At least one output node is required", nameof(settings));

            List<GraphConnection> connections = ValidateConnections(settings.Connections, nodes);
            EnsureAcyclic(nodes.Keys, connections);
            connections = Prune(nodes, connections);

            // only hidden nodes that still have a connection after pruning lie on an input-output path
            var usedNodes = new HashSet<uint>(connections.SelectMany(x => new[] { x.FromNode, x.ToNode }));
            List<GraphNode> hidden = settings.Nodes.Where(x => x.Type == NodeType.HIDDEN && usedNodes.Contains(x.Id)).ToList();

            return new Topology(inputs, biases, hidden, outputs, connections);
        }

        /// <summary>
        /// The document a built network would save, as far as <see cref="Compact.GraphPlan"/> needs it, without
        /// building neurons and synapses first
        /// </summary>
        internal static NetworkDocument ToPlanDocument(GraphNetworkSettings settings)
        {
            Topology topology = Prepare(settings);
            var document = new NetworkDocument { Kind = NetworkKind.Graph, LossType = settings.LossType, LossOptions = settings.LossOptions, OutputApplySoftMax = settings.OutputApplySoftMax };
            var indexes = new Dictionary<uint, int>();

            List<int> Add(List<GraphNode> nodes, ActivationTypes defaultActivation, bool? applyActivation)
            {
                var layer = new List<int>(nodes.Count);
                foreach (GraphNode node in nodes)
                {
                    indexes.Add(node.Id, document.Neurons.Count);
                    layer.Add(document.Neurons.Count);
                    document.Neurons.Add(new NeuronDocument
                    {
                        Id = node.Id,
                        Type = node.Type,
                        Activation = node.Type == NodeType.BIAS ? ActivationTypes.Identity : node.ActivationType ?? defaultActivation,
                        ApplyActivation = applyActivation
                    });
                }

                return layer;
            }

            document.InputLayer = Add(topology.Inputs, settings.InputActivationType, null);
            document.InputLayer.AddRange(Add(topology.Biases, ActivationTypes.Identity, null));
            document.HiddenLayers.Add(Add(topology.Hidden, settings.HiddenActivationType, null));
            document.OutputLayer = Add(topology.Outputs, settings.OutputActivationType, !settings.OutputApplySoftMax);

            foreach (GraphConnection connection in topology.Connections)
            {
                document.Synapses.Add(new SynapseDocument
                {
                    Id = connection.Id,
                    From = indexes[connection.FromNode],
                    To = indexes[connection.ToNode],
                    Weight = connection.Weight,
                    Optimizer = settings.OptimizerType
                });
            }

            return document;
        }

        internal static GraphNetwork Build(GraphNetworkSettings settings)
        {
            Topology topology = Prepare(settings);
            List<GraphNode> inputs = topology.Inputs;
            List<GraphNode> biases = topology.Biases;
            List<GraphNode> outputs = topology.Outputs;
            List<GraphConnection> connections = topology.Connections;

            IUniqueIdentityGenerator ids = settings.IdentityGenerator ?? new IncrementalGenerator();
            var network = new Network(new NetworkSettings
            {
                LossType = settings.LossType,
                LossOptions = settings.LossOptions,
                OutputApplySoftMax = settings.OutputApplySoftMax,
                ShouldNormalize = false,
                IdentityGenerator = ids,
                InputSettings = new InputLayerSettings { NeuronsCount = (ushort)inputs.Count, ActivationType = settings.InputActivationType },
                OutputSettings = new OutputLayerSettings { NeuronsCount = (ushort)outputs.Count, ActivationType = settings.OutputActivationType },
            });

            var neurons = new Dictionary<uint, INeuron>();

            var inputLayer = new Layer<IInputNeuron>();
            foreach (GraphNode node in inputs)
            {
                var neuron = new InputNeuron(node.ActivationType ?? settings.InputActivationType, id: node.Id);
                neurons.Add(node.Id, neuron);
                inputLayer.Add(neuron);
            }

            foreach (GraphNode node in biases)
            {
                var neuron = new BiasNeuron(ActivationTypes.Identity, node.Id);
                neurons.Add(node.Id, neuron);
                inputLayer.Add(neuron);
            }

            var hiddenLayer = new Layer<IHiddenNeuron>();
            var hiddenIds = new List<uint>();
            foreach (GraphNode node in topology.Hidden)
            {
                var neuron = new HiddenNeuron(node.ActivationType ?? settings.HiddenActivationType, node.Id);
                neurons.Add(node.Id, neuron);
                hiddenLayer.Add(neuron);
                hiddenIds.Add(node.Id);
            }

            var outputLayer = new Layer<IOutputNeuron>();
            foreach (GraphNode node in outputs)
            {
                var neuron = new OutputNeuron(node.ActivationType ?? settings.OutputActivationType, !settings.OutputApplySoftMax, node.Id);
                neurons.Add(node.Id, neuron);
                outputLayer.Add(neuron);
            }

            var synapses = new Dictionary<uint, ActiveSynapse>();
            // plain gradient descent keeps no state, so all synapses share one optimizer
            IOptimizer shared = NetworkBuilder.SharedOptimizer(settings.OptimizerType, network.Optimizers);
            foreach (GraphConnection connection in connections)
            {
                INeuron from = neurons[connection.FromNode];
                INeuron to = neurons[connection.ToNode];

                IOptimizer optimizer = shared;
                if (optimizer == null)
                {
                    optimizer = NetworkUtils.GetOptimizer(settings.OptimizerType);
                    if (optimizer is LearningRateOptimizer learningRateOptimizer)
                    {
                        network.Optimizers.Add(learningRateOptimizer);
                    }
                }

                var synapse = new ActiveSynapse(from, to, ids, optimizer, connection.Weight);
                to.Inputs.Add(synapse);
                from.Outputs.Add(synapse);
                synapses.Add(connection.Id, synapse);
            }

            network.InputNeurons = inputLayer;
            network.HiddenLayers = new List<ILayer<IHiddenNeuron>> { hiddenLayer };
            network.OutputNeurons = outputLayer;

            return new GraphNetwork(network, synapses, hiddenIds);
        }

        private static List<GraphConnection> ValidateConnections(IEnumerable<GraphConnection> connections, Dictionary<uint, GraphNode> nodes)
        {
            var result = new List<GraphConnection>();
            var connectionIds = new HashSet<uint>();

            foreach (GraphConnection connection in connections ?? Enumerable.Empty<GraphConnection>())
            {
                if (connection == null) throw new ArgumentException("Connections must not contain null", nameof(connections));
                if (!connectionIds.Add(connection.Id))
                    throw new ArgumentException($"Duplicate connection id {connection.Id}", nameof(connections));
                if (!connection.Enabled) continue;

                if (!nodes.TryGetValue(connection.FromNode, out GraphNode from))
                    throw new ArgumentException($"Connection {connection.Id} starts at unknown node {connection.FromNode}", nameof(connections));
                if (!nodes.TryGetValue(connection.ToNode, out GraphNode to))
                    throw new ArgumentException($"Connection {connection.Id} ends at unknown node {connection.ToNode}", nameof(connections));
                if (from.Type == NodeType.OUTPUT)
                    throw new ArgumentException($"Connection {connection.Id} starts at output node {from.Id}", nameof(connections));
                if (to.Type == NodeType.INPUT || to.Type == NodeType.BIAS)
                    throw new ArgumentException($"Connection {connection.Id} ends at input node {to.Id}", nameof(connections));

                result.Add(connection);
            }

            return result;
        }

        /// <summary>
        /// Recurrent connections would make a neuron wait forever for its own signal
        /// </summary>
        private static void EnsureAcyclic(IEnumerable<uint> nodeIds, List<GraphConnection> connections)
        {
            var inDegree = nodeIds.ToDictionary(x => x, _ => 0);
            foreach (GraphConnection connection in connections)
            {
                inDegree[connection.ToNode]++;
            }

            ILookup<uint, uint> successors = connections.ToLookup(x => x.FromNode, x => x.ToNode);
            var queue = new Queue<uint>(inDegree.Where(x => x.Value == 0).Select(x => x.Key));
            int visited = 0;

            while (queue.Count > 0)
            {
                uint node = queue.Dequeue();
                visited++;
                foreach (uint next in successors[node])
                {
                    if (--inDegree[next] == 0) queue.Enqueue(next);
                }
            }

            if (visited != inDegree.Count)
                throw new ArgumentException("The connections form a cycle; only feed-forward graphs are supported", nameof(connections));
        }

        /// <summary>
        /// Keeps only connections that lie on a path from an input (or bias) to an output. A neuron fed by a
        /// dead branch would wait forever for that branch during propagation, and a neuron feeding a dead end
        /// would wait forever for its error during backpropagation
        /// </summary>
        private static List<GraphConnection> Prune(Dictionary<uint, GraphNode> nodes, List<GraphConnection> connections)
        {
            HashSet<uint> fed = Reach(nodes.Values.Where(x => x.Type == NodeType.INPUT || x.Type == NodeType.BIAS).Select(x => x.Id),
                                      connections.ToLookup(x => x.FromNode, x => x.ToNode));
            HashSet<uint> useful = Reach(nodes.Values.Where(x => x.Type == NodeType.OUTPUT).Select(x => x.Id),
                                         connections.ToLookup(x => x.ToNode, x => x.FromNode));

            return connections.Where(x => fed.Contains(x.FromNode) && useful.Contains(x.ToNode)).ToList();
        }

        private static HashSet<uint> Reach(IEnumerable<uint> start, ILookup<uint, uint> edges)
        {
            var reached = new HashSet<uint>(start);
            var stack = new Stack<uint>(reached);

            while (stack.Count > 0)
            {
                foreach (uint next in edges[stack.Pop()])
                {
                    if (reached.Add(next)) stack.Push(next);
                }
            }

            return reached;
        }
    }
}
