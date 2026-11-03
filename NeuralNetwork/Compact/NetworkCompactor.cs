using System;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Compact;
using ANN.Net.Convolutional;
using ANN.Net.Serialization;

namespace ANN.Net
{
    /// <summary>
    /// Shrinks trained networks for inference: the neurons, synapses, layers, optimizers and delegates of a network are
    /// replaced by flat arrays of weights
    /// </summary>
    public static class NetworkCompactor
    {
        /// <summary>
        /// Inference-only copy of a network from <c>NetworkFactory.BuildFFN</c> (without recurrent connections),
        /// <c>BuildGraph</c> or <c>BuildConvolutional</c>. The copy computes what the network computes (convolutional
        /// networks as with <see cref="IConvolutionalNetwork.Training"/> set to false); later training of the network
        /// does not change it. Once the original network is no longer referenced, its memory is freed
        /// </summary>
        public static ICompactNetwork Compact(INetwork network)
        {
            ArgumentNullException.ThrowIfNull(network);

            if (network is ConvolutionalNetwork convolutional)
                return new CompactConvolutionalNetwork(convolutional);

            return new CompactGraphNetwork(GraphPlan.Create(NetworkSerializer.ToDocument(network)));
        }
    }
}
