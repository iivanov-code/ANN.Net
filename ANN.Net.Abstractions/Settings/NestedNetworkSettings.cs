using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.Abstractions.Settings
{
    /// <summary>
    /// A network of networks. Node <see cref="InputNode"/> is the input vector of the whole network; every other node
    /// holds a network, receives the vectors of its <see cref="NestedNode.Inputs"/> and sends its output vector on.
    /// The nodes must not form a cycle
    /// </summary>
    public class NestedNetworkSettings
    {
        /// <summary>
        /// Id of the input vector of the whole network, usable among the inputs of every node
        /// </summary>
        public const uint InputNode = 0;

        public int InputSize { get; set; }

        public List<NestedNode> Nodes { get; set; } = new List<NestedNode>();

        /// <summary>
        /// Nodes whose output vectors, one after the other, are the output of the whole network
        /// </summary>
        public List<uint> Outputs { get; set; } = new List<uint>();

        /// <summary>
        /// Loss reported by <see cref="INetwork.Backpropagate"/>; the gradient is always output - target
        /// </summary>
        public LossFunctionTypes LossType { get; set; } = LossFunctionTypes.Squared;
    }

    public class NestedNode
    {
        public NestedNode()
        {
        }

        public NestedNode(uint id, IVectorNetwork network, params uint[] inputs)
        {
            Id = id;
            Network = network;
            Inputs = new List<uint>(inputs ?? System.Array.Empty<uint>());
        }

        /// <summary>
        /// Id of the node, not <see cref="NestedNetworkSettings.InputNode"/>
        /// </summary>
        public uint Id { get; set; }

        /// <summary>
        /// The network inside the node; a network can be inside one node only
        /// </summary>
        public IVectorNetwork Network { get; set; }

        /// <summary>
        /// Nodes whose output vectors this node receives, <see cref="NestedNetworkSettings.InputNode"/> for the input of
        /// the whole network
        /// </summary>
        public List<uint> Inputs { get; set; } = new List<uint>();

        public NodeCombination Combination { get; set; } = NodeCombination.Concatenate;
    }
}
