using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Collections;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Abstractions.Interfaces.Synapses;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Connections;
using ANN.Net.Neurons;
using ANN.Net.Utils;

namespace ANN.Net
{
    public class NetworkBuilder
    {
        private NetworkBuilder()
        { }

        public static NetworkBuilder Create()
        {
            return new NetworkBuilder();
        }

        public IList<ILayer<IHiddenNeuron>> BuildHiddenLayers(ILayer<INeuron> prevLayer, RecurrentLayerSettings settings, ICollection<LearningRateOptimizer> optimizers)
        {
            return BuildHiddenLayers(prevLayer, new RecurrentLayerSettings[] { settings }, optimizers);
        }

        public IList<ILayer<IHiddenNeuron>> BuildHiddenLayers(ILayer<INeuron> prevLayer, RecurrentLayerSettings[] settings, ICollection<LearningRateOptimizer> optimizers)
        {
            if (optimizers == null) throw new ArgumentNullException(nameof(optimizers));

            var listOfLayers = new List<ILayer<IHiddenNeuron>>();

            foreach (var layerSettings in settings)
            {
                for (int i = 0; i < layerSettings.LayersCount; i++)
                {
                    Func<RecurrentLayerSettings, IHiddenNeuron> neuronConstructor = null;
                    switch (layerSettings.CellType)
                    {
                        case NeuronType.Normal:
                        case NeuronType.RNN:
                            neuronConstructor = (settings) => new HiddenNeuron(settings.ActivationType);
                            break;

                        case NeuronType.LSTM:
                            neuronConstructor = (settings) => new OldNetLSTMCell(2, 2);
                            break;

                        case NeuronType.LSTMPeepHole:
                            neuronConstructor = (settings) => new OldNetLSTMCell(2, 2);
                            break;

                        case NeuronType.GRU:
                            neuronConstructor = (settings) => new GRUCell(settings.ActivationType);
                            break;

                        default:
                            break;
                    }

                    ILayer<IHiddenNeuron> newLayer;

                    switch (layerSettings.PrevLayerConnectionType)
                    {
                        case LayerConnectionType.FullyConnected:
                            newLayer = FullyConnectLayers<IHiddenNeuron>(prevLayer, layerSettings, optimizers, () => neuronConstructor(layerSettings));
                            break;

                        case LayerConnectionType.Convolutional:
                            newLayer = ConvolutionalConnectLayers<IHiddenNeuron>(prevLayer, layerSettings, optimizers, () => neuronConstructor(layerSettings));
                            break;

                        case LayerConnectionType.Deconvolutional:
                            newLayer = DeconvolutionalConnectLayers<IHiddenNeuron>(prevLayer, layerSettings, optimizers, () => neuronConstructor(layerSettings));
                            break;

                        case LayerConnectionType.Straight:
                            newLayer = StraightConnectLayer<IHiddenNeuron>(prevLayer, layerSettings, optimizers, () => neuronConstructor(layerSettings));
                            break;

                        default:
                            throw new ArgumentException("Invalid layer connection type");
                    }

                    if (layerSettings.CellType != NeuronType.Normal)
                    {
                        RecurrentConnectLayer<IHiddenNeuron>(newLayer, layerSettings, optimizers);
                    }

                    listOfLayers.Add(newLayer);
                    prevLayer = newLayer.AsLayerOf<INeuron>();
                }
            }

            return listOfLayers;
        }

        public ILayer<IInputNeuron> BuildInputNeurons(InputLayerSettings settings, ushort recurrentInputs = 0)
        {
            var tempInputNeurons = new Layer<IInputNeuron>(settings.NeuronsCount);

            for (int i = 0; i < settings.NeuronsCount; i++)
            {
                tempInputNeurons.Add(new InputNeuron(settings.ActivationType, recurrentInputs));
            }

            if (settings.HasBiasNeuron)
                tempInputNeurons.Add(new BiasNeuron(settings.ActivationType));

            return tempInputNeurons;
        }

        public ILayer<IOutputNeuron> BuildOutputNeurons(IEnumerable<INeuron> prevLayer,
                                                        OutputLayerSettings settings,
                                                        bool shouldActivate,
                                                        ICollection<LearningRateOptimizer> optimizers)
        {
            return FullyConnectLayers<IOutputNeuron>(prevLayer, settings, optimizers, () => new OutputNeuron(settings.ActivationType, shouldActivate));
        }

        public IInputNeuron BuildInputNeuron(ActivationTypes activationType, uint id)
        {
            return new InputNeuron(activationType, id: id);
        }

        public IOutputNeuron BuildOutputNeuron(ActivationTypes activationType, bool shouldActivate, uint id)
        {
            return new OutputNeuron(activationType, shouldActivate, id);
        }

        public IHiddenNeuron BuildHiddenNeuron(ActivationTypes activationType, uint id)
        {
            return new HiddenNeuron(activationType, id);
        }

        public IOptimizer ConnectAxon(INeuron input,
                                      INeuron output,
                                      CommonLayerSettings settings,
                                      ICollection<LearningRateOptimizer> optimizers,
                                      bool isConnectionActive = true,
                                      bool recurrentConnection = false)
        {
            Quad initialWeight;
            switch (settings.InitializationType)
            {
                case WeightInitType.Random:
                    initialWeight = NetworkUtils.RandomWeight();
                    break;

                case WeightInitType.Xavier:
                    initialWeight = NetworkUtils.XavierWeight(settings.NeuronsCount);
                    break;

                default:
                    throw new ArgumentException("Invalid value", nameof(settings.InitializationType));
            }

            IOptimizer optimizer = NetworkUtils.GetOptimizer(settings.OptimizerType);

            if (optimizer is LearningRateOptimizer)
            {
                optimizers.Add(optimizer as LearningRateOptimizer);
            }

            ISynapse axon;
            if (isConnectionActive)
            {
                axon = new ActiveSynapse(input, output, optimizer, initialWeight);
            }
            else
            {
                axon = new InactiveSynapse(input, output);
            }

            if (recurrentConnection)
            {
                output.Inputs.AddRecurrent(axon);
                input.Outputs.AddRecurrent(axon);
            }
            else
            {
                output.Inputs.Add(axon);
                input.Outputs.Add(axon);
            }

            return optimizer;
        }

        private ILayer<T> ConvolutionalConnectLayers<T>(ILayer<INeuron> prevLayer, CommonLayerSettings settings, ICollection<LearningRateOptimizer> optimizers, Func<T> neuronConstructor)
                       where T : class, INeuron
        {
            var layer = new Layer<T>(prevLayer.Count - 1);

            for (int i = 1; i < prevLayer.Count - 1; i++)
            {
                var neuron = neuronConstructor();
                layer.Add(neuron);

                for (int j = i - 1; j < i + 1; j++)
                {
                    ConnectAxon(prevLayer[i], neuron, settings, optimizers);
                }
            }

            ConnectAxon(prevLayer[0], layer[0], settings, optimizers);
            ConnectAxon(prevLayer[prevLayer.Count - 1], layer[layer.Count - 1], settings, optimizers);

            return layer;
        }

        private ILayer<T> DeconvolutionalConnectLayers<T>(ILayer<INeuron> prevLayer,
                                                          CommonLayerSettings settings,
                                                          ICollection<LearningRateOptimizer> optimizers,
                                                          Func<T> neuronConstructor)
                       where T : class, INeuron
        {
            var layer = new Layer<T>(prevLayer.Count + 1);

            for (int i = 0; i < prevLayer.Count; i++)
            {
                var neuron = neuronConstructor();
                layer.Add(neuron);

                for (int j = i; j < i + 2; j++)
                {
                    ConnectAxon(prevLayer[i], neuron, settings, optimizers);
                }
            }

            return layer;
        }

        private ILayer<T> StraightConnectLayer<T>(ILayer<INeuron> prevLayer,
                                                  CommonLayerSettings settings,
                                                  ICollection<LearningRateOptimizer> optimizers,
                                                  Func<T> neuronConstructor)
               where T : class, INeuron
        {
            if (settings.NeuronsCount != prevLayer.Count)
                throw new ArgumentException("Бройката неврони в двата слоя трябва да е еднаква");

            var layer = new Layer<T>(prevLayer.Count + 1);

            for (int i = 0; i < prevLayer.Count; i++)
            {
                var neuron = neuronConstructor();
                layer.Add(neuron);
                ConnectAxon(prevLayer[i], neuron, settings, optimizers);
            }

            return layer;
        }

        private ILayer<T> FullyConnectLayers<T>(IEnumerable<INeuron> prevLayer, CommonLayerSettings settings, ICollection<LearningRateOptimizer> optimizers, Func<T> neuronConstructor)
             where T : class, INeuron
        {
            var layer = new Layer<T>(settings.NeuronsCount);

            for (int i = 0; i < settings.NeuronsCount; i++)
            {
                var neuron = neuronConstructor();
                layer.Add(neuron);
                foreach (var inNeuron in prevLayer)
                {
                    ConnectAxon(inNeuron, neuron, settings, optimizers);
                }
            }

            if (settings.HasBiasNeuron)
                layer.Add(new BiasNeuron(settings.ActivationType) as T);

            return layer;
        }

        private void RecurrentConnectLayer<T>(ILayer<T> layer, RecurrentLayerSettings settings, ICollection<LearningRateOptimizer> optimizers)
                                             where T : class, INeuron
        {
            if (layer == null || layer.Count == 0)
                throw new ArgumentNullException(nameof(layer), "Layer must not be null or empty");

            if ((settings.Direction & RecurrentDirection.Forward) == RecurrentDirection.Forward)
            {
                for (int i = 0; i < layer.Count - 2; i++)
                {
                    if (settings.FullyConnectLayer)
                    {
                        for (int j = i + 1; j < layer.Count - 2; j++)
                        {
                            ConnectAxon(layer[i], layer[j], settings, optimizers, true);
                        }
                    }
                    else
                    {
                        ConnectAxon(layer[i], layer[i + 1], settings, optimizers, true);
                    }

                    if (settings.HasSelfConnection)
                    {
                        ConnectAxon(layer[i], layer[i], settings, optimizers, true);
                    }
                }
            }

            if ((settings.Direction & RecurrentDirection.Backward) == RecurrentDirection.Backward)
            {
                for (int i = layer.Count - 1; i > 0; i--)
                {
                    if (settings.FullyConnectLayer)
                    {
                        for (int j = i - 1; j >= 0; j++)
                        {
                            ConnectAxon(layer[i], layer[j], settings, optimizers, true);
                        }
                    }
                    else
                    {
                        ConnectAxon(layer[i], layer[i - 1], settings, optimizers, true);
                    }

                    if (settings.HasSelfConnection && (settings.Direction & RecurrentDirection.Forward) != RecurrentDirection.Forward)
                    {
                        ConnectAxon(layer[i], layer[i], settings, optimizers, true);
                    }
                }
            }
        }
    }
}
