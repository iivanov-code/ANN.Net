using System.Collections.Generic;

namespace ANN.Net.Abstractions.Interfaces
{
    /// <summary>
    /// Network of networks: every node holds a whole network that takes a vector and gives a vector, built by
    /// <c>NetworkFactory.BuildNested</c>. A nested network is a vector network itself, so it can sit inside a node of
    /// another one, to any depth
    /// </summary>
    public interface INestedNetwork : INetwork, IVectorNetwork
    {
        /// <summary>
        /// Outputs for the input values; one method for <see cref="INetwork"/> and <see cref="IVectorNetwork"/>
        /// </summary>
        new Quad[] Propagate(params Quad[] values);

        /// <summary>
        /// The network inside every node, by node id
        /// </summary>
        IReadOnlyDictionary<uint, IVectorNetwork> Nodes { get; }

        /// <summary>
        /// Output vector of every node from the last propagation
        /// </summary>
        IReadOnlyDictionary<uint, Quad[]> NodeValues { get; }
    }
}
