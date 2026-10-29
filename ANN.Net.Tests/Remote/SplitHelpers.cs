using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Connections;
using ANN.Net.IdentityGenerators;
using ANN.Net.Remote;

namespace ANN.Net.Tests.Remote
{
    internal static class SplitHelpers
    {
        public static NetworkSettings Whole(ushort inputs,
                                            ushort outputs,
                                            ActivationTypes outputActivation = ActivationTypes.Identity,
                                            bool inputBias = false)
        {
            return new NetworkSettings
            {
                IdentityGenerator = new IncrementalGenerator(),
                LossType = LossFunctionTypes.Squared,
                InputSettings = new InputLayerSettings { NeuronsCount = inputs, ActivationType = ActivationTypes.Identity, HasBiasNeuron = inputBias, InitializationType = WeightInitType.Xavier },
                OutputSettings = new OutputLayerSettings { NeuronsCount = outputs, ActivationType = outputActivation, HasBiasNeuron = false, InitializationType = WeightInitType.Xavier }
            };
        }

        public static RecurrentLayerSettings Layer(ushort count, ActivationTypes activation, bool bias = false, ushort layersCount = 1)
        {
            return new RecurrentLayerSettings
            {
                NeuronsCount = count,
                LayersCount = layersCount,
                CellType = NeuronType.Normal,
                PrevLayerConnectionType = LayerConnectionType.FullyConnected,
                ActivationType = activation,
                HasBiasNeuron = bias,
                InitializationType = WeightInitType.Xavier
            };
        }

        /// <summary>
        /// Head and tail wired together in this process, without sockets
        /// </summary>
        public static SplitNetwork BuildInProcess(NetworkSettings whole, RecurrentLayerSettings[] layers, int boundary, out Network tail)
        {
            tail = (Network)RemoteNetworkFactory.BuildTail(whole, layers, boundary);
            SynapseConnection connection = new SynapseConnection(RemoteNetworkPart.Create(tail));
            return RemoteNetworkFactory.BuildHead(whole, layers, boundary, connection);
        }

        /// <summary>
        /// Every weighted synapse of a network: the outputs of its input and hidden neurons, remote ones excluded
        /// </summary>
        public static IEnumerable<BaseSynapse> TrainableSynapses(Network network)
        {
            IEnumerable<INeuron> neurons = network.InputNeurons.Cast<INeuron>();
            if (network.HiddenLayers != null)
            {
                neurons = neurons.Concat(network.HiddenLayers.SelectMany(layer => layer.Cast<INeuron>()));
            }

            return neurons.SelectMany(n => n.Outputs.OfType<BaseSynapse>()).Where(s => !(s is RemoteSynapse));
        }

        public static double Loss(Quad[] values, Quad[] targets)
        {
            double loss = 0;
            for (int i = 0; i < values.Length; i++)
            {
                double difference = (float)values[i] - (float)targets[i];
                loss += difference * difference / 2;
            }

            return loss;
        }
    }
}
