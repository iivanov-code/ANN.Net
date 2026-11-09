using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Collections;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Abstractions.Spiking;
using ANN.Net.Embeddings;
using ANN.Net.LanguageModels;
using ANN.Net.Tokenizers;
using ANN.Net.Transformers;

namespace ANN.Net
{
    public static class NetworkFactory
    {
        public static INetwork BuildFFN(NetworkSettings settings, RecurrentLayerSettings hiddenSettings = null)
        {
            return BuildFFN(settings, null, hiddenSettings);
        }

        internal static INetwork BuildFFN(NetworkSettings settings, IEnumerable<INeuron> additionalInputs = null, RecurrentLayerSettings hiddenSettings = null)
        {
            return BuildFFN(settings, additionalInputs, hiddenSettings == null ? null : new[] { hiddenSettings });
        }

        /// <summary>
        /// Builds a fully connected network with any number of hidden layer descriptions in order
        /// </summary>
        internal static INetwork BuildFFN(NetworkSettings settings, IEnumerable<INeuron> additionalInputs, RecurrentLayerSettings[] hiddenSettings)
        {
            using System.IDisposable seed = Utils.NetworkUtils.SeedRandom(settings.Seed);
            Network ffn = new Network(settings);
            NetworkBuilder builder = NetworkBuilder.Create();

            ffn.InputNeurons = builder.BuildInputNeurons(settings.InputSettings);
            ILayer<INeuron> prevLayer;
            if (hiddenSettings != null && hiddenSettings.Length > 0)
            {
                ffn.HiddenLayers = builder.BuildHiddenLayers(ffn.InputNeurons.AsLayerOf<INeuron>(),
                                                             settings.IdentityGenerator,
                                                             hiddenSettings,
                                                             ffn.Optimizers);
            }

            if (ffn.HiddenLayers != null && ffn.HiddenLayers.Count > 0)
            {
                prevLayer = ffn.HiddenLayers.Last().AsLayerOf<INeuron>();
            }
            else
            {
                prevLayer = ffn.InputNeurons.AsLayerOf<INeuron>();
            }

            if (additionalInputs != null)
                prevLayer = new Layer<INeuron>(prevLayer.Concat(additionalInputs));

            ffn.OutputNeurons = builder.BuildOutputNeurons(prevLayer,
                                                           settings.IdentityGenerator,
                                                           settings.OutputSettings,
                                                           !settings.OutputApplySoftMax,
                                                           ffn.Optimizers);
            return ffn;
        }

        /// <summary>
        /// Builds the fully connected network <see cref="BuildFFN(NetworkSettings, RecurrentLayerSettings)"/> builds,
        /// with one weight matrix per layer instead of neuron and synapse objects. With the same settings and
        /// <see cref="NetworkSettings.Seed"/> it starts from the same weights and computes and trains bit-identically,
        /// using a fraction of the memory and time. Hidden layers must be fully connected layers of normal neurons
        /// </summary>
        public static IMatrixNetwork BuildMatrixFFN(NetworkSettings settings, params RecurrentLayerSettings[] hiddenSettings)
        {
            return Matrix.MatrixNetwork.Build(settings, hiddenSettings);
        }

        /// <summary>
        /// Converts a fully connected network of <see cref="BuildFFN(NetworkSettings, RecurrentLayerSettings)"/> into a
        /// matrix network with the same weights, optimizer types and learning rates. The state of optimizers such as Adam
        /// starts afresh
        /// </summary>
        public static IMatrixNetwork ToMatrix(INetwork network)
        {
            ArgumentNullException.ThrowIfNull(network);

            return network switch
            {
                IMatrixNetwork matrix => matrix,
                Network layered => Matrix.MatrixNetwork.FromNetwork(layered),
                _ => throw new NotSupportedException($"A {network.GetType().Name} cannot be converted to a matrix network")
            };
        }

        /// <summary>
        /// Builds a network of networks: every node holds a whole network (matrix, neuron, graph, convolutional,
        /// transformer through <see cref="AsVectorNetwork(ISequenceNetwork, int)"/>, or another nested network) that
        /// takes a vector and gives a vector. Backpropagation trains the networks of all nodes together
        /// </summary>
        public static INestedNetwork BuildNested(NestedNetworkSettings settings)
        {
            return Nested.NestedNetwork.Build(settings);
        }

        /// <summary>
        /// The network as a vector network for a node of a nested network. Networks of <see cref="BuildFFN(NetworkSettings, RecurrentLayerSettings)"/>,
        /// <see cref="BuildMatrixFFN"/>, <see cref="BuildGraph"/>, <see cref="BuildConvolutional"/> and <see cref="BuildNested"/> are vector networks
        /// </summary>
        public static IVectorNetwork AsVectorNetwork(INetwork network)
        {
            ArgumentNullException.ThrowIfNull(network);
            return network as IVectorNetwork ?? throw new NotSupportedException($"A {network.GetType().Name} cannot pass gradients and cannot be inside a nested network");
        }

        /// <summary>
        /// A transformer as a vector network for sequences of <paramref name="sequenceLength"/> tokens: the vectors in
        /// and out hold the tokens one after the other
        /// </summary>
        public static IVectorNetwork AsVectorNetwork(ISequenceNetwork network, int sequenceLength)
        {
            ArgumentNullException.ThrowIfNull(network);
            if (network is not TransformerNetwork transformer) throw new NotSupportedException($"A {network.GetType().Name} cannot be inside a nested network");
            return new Nested.SequenceVectorNetwork(transformer, sequenceLength);
        }

        /// <summary>
        /// Builds a feed-forward network with an arbitrary acyclic topology, such as the phenotype of a NEAT genome.
        /// Disabled connections and every neuron that is not on a path from an input to an output are left out
        /// </summary>
        public static IGraphNetwork BuildGraph(GraphNetworkSettings settings)
        {
            return GraphNetwork.Build(settings);
        }

        /// <summary>
        /// Builds a convolutional network: convolution, pooling and dense layers followed by a dense output layer
        /// </summary>
        public static IConvolutionalNetwork BuildConvolutional(ConvolutionalNetworkSettings settings)
        {
            return new Convolutional.ConvolutionalNetwork(settings);
        }

        /// <summary>
        /// Builds a layered spiking network, for instance to be trained without labels with
        /// <see cref="ISpikingNetwork.Learn"/> (spike-timing-dependent plasticity). To turn a trained network into a
        /// spiking one, use <see cref="SpikingNetworkConverter"/>
        /// </summary>
        public static ISpikingNetwork BuildSpiking(SpikingNetworkSettings settings)
        {
            return Spiking.SpikingBuilder.Build(settings);
        }

        /// <summary>
        /// Builds a spiking network of any topology from its layers and synapses, such as the phenotype of a NEAT genome
        /// </summary>
        public static ISpikingNetwork BuildSpiking(SpikingGraphSettings settings)
        {
            return Spiking.SpikingBuilder.Build(settings);
        }

        public static INetwork BuildRNN(NetworkSettings settings, HiddenLayerSettings hiddenSettings)
        {
            Network rnn = new Network(settings);
            NetworkBuilder builder = NetworkBuilder.Create();
            return null;
        }

        public static INetwork BuildCellRNN(NetworkSettings settings, RecurrentLayerSettings[] cellSettings)
        {
            return null;
        }

        public static ISequenceNetwork BuildTransformer(TransformerSettings settings)
        {
            return new TransformerNetwork(settings);
        }

        public static ITokenizer BuildBpeTokenizer(BpeTokenizerSettings settings)
        {
            return new BpeTokenizer(settings);
        }

        public static IEmbeddingNetwork BuildWord2Vec(Word2VecSettings settings)
        {
            return new Word2VecNetwork(settings);
        }

        /// <summary>
        /// Ties a trained tokenizer, optional token embeddings and a soft-max sequence network
        /// into a next-token language model
        /// </summary>
        public static ILanguageModel BuildLanguageModel(LanguageModelSettings settings,
                                                        ITokenizer tokenizer,
                                                        ISequenceNetwork network,
                                                        IEmbeddingNetwork embeddings = null)
        {
            return new LanguageModel(settings, tokenizer, network, embeddings);
        }
    }
}
