using System;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Settings;
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
        /// <param name="options">Number format of the weights; 32-bit floats when null</param>
        public static ICompactNetwork Compact(INetwork network, CompactOptions options = null)
        {
            ArgumentNullException.ThrowIfNull(network);

            if (network is ConvolutionalNetwork convolutional)
                return new CompactConvolutionalNetwork(convolutional, options);

            if (network is Matrix.MatrixNetwork matrix)
                return new CompactGraphNetwork(GraphPlan.Create(matrix), options);

            return new CompactGraphNetwork(GraphPlan.Create(NetworkSerializer.ToDocument(network)), options);
        }

        /// <summary>
        /// Inference-only network of a graph, computing what <c>NetworkFactory.BuildGraph(settings)</c> would compute
        /// but without building the neurons and synapses first: for networks that are only evaluated, such as the
        /// phenotypes of NEAT genomes, this is several times faster and smaller. The settings are validated and pruned
        /// as <c>BuildGraph</c> does
        /// </summary>
        /// <summary>
        /// Inference-only copy of a transformer from <c>NetworkFactory.BuildTransformer</c> (also the one inside a
        /// language model). With 32-bit weights it computes what the transformer computes; the options choose the number
        /// format of every projection and prune them as for other networks
        /// </summary>
        public static ICompactSequenceNetwork CompactSequence(ISequenceNetwork network, CompactOptions options = null)
        {
            ArgumentNullException.ThrowIfNull(network);

            if (network is Transformers.TransformerNetwork transformer) return new CompactTransformer(transformer, options);

            throw new NotSupportedException($"A {network.GetType().Name} cannot be compacted");
        }

        public static ICompactNetwork CompactGraph(GraphNetworkSettings settings, CompactOptions options = null)
        {
            return new CompactGraphNetwork(GraphPlan.Create(GraphNetwork.ToPlanDocument(settings)), options);
        }
    }
}
