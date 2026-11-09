using System;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Abstractions.Spiking;
using ANN.Net.Compact;
using ANN.Net.Serialization;

namespace ANN.Net
{
    /// <summary>
    /// Turns trained networks into spiking networks: train with ReLU, leaky ReLU, the identity, sigmoid or tanh as
    /// usual, then convert. Every neuron becomes a spiking neuron whose firing rate follows its activation, so the
    /// spiking network computes approximately what the trained one computes, more closely the more time steps it runs
    /// </summary>
    public static class SpikingNetworkConverter
    {
        /// <summary>
        /// Spiking copy of a network from <c>NetworkFactory.BuildFFN</c> (without recurrent connections),
        /// <c>BuildMatrixFFN</c> or <c>BuildGraph</c>. Later training of the network does not change the copy
        /// </summary>
        public static ISpikingNetwork Convert(INetwork network, SpikingConversionOptions options)
        {
            ArgumentNullException.ThrowIfNull(network);

            if (network is Convolutional.ConvolutionalNetwork) throw new NotSupportedException("Convolutional networks cannot be converted to spiking networks yet");
            if (network is Nested.NestedNetwork) throw new NotSupportedException("Nested networks cannot be converted to spiking networks yet");

            GraphPlan plan = network is Matrix.MatrixNetwork matrix ? GraphPlan.Create(matrix) : GraphPlan.Create(NetworkSerializer.ToDocument(network));
            return Spiking.SpikingConversion.Convert(plan, options);
        }

        /// <summary>
        /// Spiking network of a graph, such as the phenotype of a NEAT genome, computing approximately what
        /// <c>NetworkFactory.BuildGraph(settings)</c> computes, without building the neurons and synapses first
        /// </summary>
        public static ISpikingNetwork ConvertGraph(GraphNetworkSettings settings, SpikingConversionOptions options)
        {
            return Spiking.SpikingConversion.Convert(GraphPlan.Create(GraphNetwork.ToPlanDocument(settings)), options);
        }
    }
}
