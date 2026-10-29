using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Collections;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Abstractions.Settings;
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
