using System.Collections.Generic;
using ANN.Net.Abstractions.Optimizers;

namespace ANN.Net.Abstractions.Interfaces
{
    /// <summary>
    /// Network built from a free-form graph description (see <c>NetworkFactory.BuildGraph</c>)
    /// </summary>
    public interface IGraphNetwork : INetwork, IVectorNetwork
    {
        /// <summary>
        /// Outputs for the input values; one method for <see cref="INetwork"/> and <see cref="IVectorNetwork"/>
        /// </summary>
        new Quad[] Propagate(params Quad[] values);

        /// <summary>
        /// Optimizers of every synapse, to set the learning rate before calling <see cref="INetwork.Backpropagate"/>
        /// </summary>
        new ICollection<LearningRateOptimizer> Optimizers { get; }

        /// <summary>
        /// Current weight of every connection that made it into the network, keyed by connection id.
        /// Disabled connections and connections that cannot carry a signal from an input to an output are left out
        /// </summary>
        IReadOnlyDictionary<uint, Quad> ConnectionWeights { get; }

        /// <summary>
        /// Ids of the hidden nodes that made it into the network
        /// </summary>
        IReadOnlyCollection<uint> HiddenNodeIds { get; }
    }
}
