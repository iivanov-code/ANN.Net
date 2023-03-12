using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Collections;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Abstractions.Settings;

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
            Network ffn = new Network(settings);
            NetworkBuilder builder = NetworkBuilder.Create();

            ffn.InputNeurons = builder.BuildInputNeurons(settings.InputSettings);
            ILayer<INeuron> prevLayer;
            if (hiddenSettings != null)
            {
                ffn.HiddenLayers = builder.BuildHiddenLayers(ffn.InputNeurons.AsLayerOf<INeuron>(),
                                                             settings.IdentityGenerator,
                                                             hiddenSettings,
                                                             ffn.Optimizers);
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
    }
}
