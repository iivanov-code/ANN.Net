using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.Abstractions.Settings
{
    /// <summary>
    /// One neuron of a network with a free-form topology
    /// </summary>
    public sealed class GraphNode
    {
        public GraphNode()
        { }

        public GraphNode(uint id, NodeType type, ActivationTypes? activationType = null)
        {
            Id = id;
            Type = type;
            ActivationType = activationType;
        }

        /// <summary>
        /// Unique identity of the node inside the graph; becomes the neuron <c>ID</c>
        /// </summary>
        public uint Id { get; set; }

        public NodeType Type { get; set; }

        /// <summary>
        /// Overrides the activation chosen in <see cref="GraphNetworkSettings"/> for this node type
        /// </summary>
        public ActivationTypes? ActivationType { get; set; }
    }

    /// <summary>
    /// One weighted connection of a network with a free-form topology
    /// </summary>
    public sealed class GraphConnection
    {
        public GraphConnection()
        { }

        public GraphConnection(uint id, uint fromNode, uint toNode, Quad weight, bool enabled = true)
        {
            Id = id;
            FromNode = fromNode;
            ToNode = toNode;
            Weight = weight;
            Enabled = enabled;
        }

        /// <summary>
        /// Unique identity of the connection, used to read the weight back after training
        /// </summary>
        public uint Id { get; set; }

        public uint FromNode { get; set; }

        public uint ToNode { get; set; }

        public Quad Weight { get; set; }

        /// <summary>
        /// Disabled connections are left out of the network
        /// </summary>
        public bool Enabled { get; set; } = true;
    }

    /// <summary>
    /// Description of a feed-forward network with an arbitrary acyclic topology, for example the phenotype of a NEAT genome.
    /// Inputs are fed and outputs are returned in the order their nodes appear in <see cref="Nodes"/>
    /// </summary>
    public class GraphNetworkSettings
    {
        public IList<GraphNode> Nodes { get; set; } = new List<GraphNode>();

        public IList<GraphConnection> Connections { get; set; } = new List<GraphConnection>();

        public ActivationTypes InputActivationType { get; set; } = ActivationTypes.Identity;

        public ActivationTypes HiddenActivationType { get; set; } = ActivationTypes.HyperbolicTangens;

        public ActivationTypes OutputActivationType { get; set; } = ActivationTypes.Sigmoid;

        public OptimizerTypes OptimizerType { get; set; } = OptimizerTypes.Normal;

        public LossFunctionTypes LossType { get; set; } = LossFunctionTypes.Squared;

        public bool OutputApplySoftMax { get; set; }

        /// <summary>
        /// Generator for the synapse identities; a new incremental generator is used when not set
        /// </summary>
        public IUniqueIdentityGenerator IdentityGenerator { get; set; }
    }
}
