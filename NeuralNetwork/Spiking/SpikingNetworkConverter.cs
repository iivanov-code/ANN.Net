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
            return Spiking.SpikingConversion.Convert(Plan(network), options);
        }

        /// <summary>
        /// Time-to-first-spike copy of a network from <c>NetworkFactory.BuildFFN</c> (without recurrent connections),
        /// <c>BuildMatrixFFN</c> or <c>BuildGraph</c>: every neuron fires at most once, earlier for a larger value, so
        /// the work is at most one multiply-add per synapse. <see cref="SpikingConversionOptions.TimeSteps"/> is the
        /// window of every layer. Neurons of any activation can be converted
        /// </summary>
        public static IFirstSpikeNetwork ConvertToFirstSpike(INetwork network, SpikingConversionOptions options)
        {
            return Spiking.FirstSpikeNetwork.Convert(Plan(network), options);
        }

        /// <summary>
        /// Time-to-first-spike network of a graph, such as the phenotype of a NEAT genome
        /// </summary>
        public static IFirstSpikeNetwork ConvertGraphToFirstSpike(GraphNetworkSettings settings, SpikingConversionOptions options)
        {
            return Spiking.FirstSpikeNetwork.Convert(GraphPlan.Create(GraphNetwork.ToPlanDocument(settings)), options);
        }

        private static GraphPlan Plan(INetwork network)
        {
            ArgumentNullException.ThrowIfNull(network);

            if (network is Convolutional.ConvolutionalNetwork) throw new NotSupportedException("Convolutional networks cannot be converted to spiking networks yet");
            if (network is Nested.NestedNetwork) throw new NotSupportedException("Nested networks cannot be converted to spiking networks yet");

            return network is Matrix.MatrixNetwork matrix ? GraphPlan.Create(matrix) : GraphPlan.Create(NetworkSerializer.ToDocument(network));
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
