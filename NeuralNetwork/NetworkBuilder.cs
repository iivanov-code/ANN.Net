using System;
using System.Collections.Generic;
using System.Linq;
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

        public IList<ILayer<IHiddenNeuron>> BuildHiddenLayers(ILayer<INeuron> prevLayer, IUniqueIdentityGenerator identityGenerator, RecurrentLayerSettings settings, ICollection<LearningRateOptimizer> optimizers)
        {
            return BuildHiddenLayers(prevLayer, identityGenerator, new RecurrentLayerSettings[] { settings }, optimizers);
        }

        public IList<ILayer<IHiddenNeuron>> BuildHiddenLayers(ILayer<INeuron> prevLayer,
                                                              IUniqueIdentityGenerator identityGenerator,
                                                              RecurrentLayerSettings[] settings,
                                                              ICollection<LearningRateOptimizer> optimizers)
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
                            neuronConstructor = (settings) => new LSTMCell(2, 2);
                            break;

                        case NeuronType.LSTMPeepHole:
                            neuronConstructor = (settings) => new LSTMCell(2, 2);
                            break;

                        case NeuronType.GRU:
                            neuronConstructor = (settings) => new GRUCell(2, 2);
                            break;

                        default:
                            break;
                    }

                    ILayer<IHiddenNeuron> newLayer;

                    switch (layerSettings.PrevLayerConnectionType)
                    {
                        case LayerConnectionType.FullyConnected:
                            newLayer = FullyConnectLayers<IHiddenNeuron>(prevLayer, identityGenerator, layerSettings, optimizers, () => neuronConstructor(layerSettings));
                            break;

                        case LayerConnectionType.Convolutional:
                            newLayer = ConvolutionalConnectLayers<IHiddenNeuron>(prevLayer, identityGenerator, layerSettings, optimizers, () => neuronConstructor(layerSettings));
                            break;

                        case LayerConnectionType.Deconvolutional:
                            newLayer = DeconvolutionalConnectLayers<IHiddenNeuron>(prevLayer, identityGenerator, layerSettings, optimizers, () => neuronConstructor(layerSettings));
                            break;

                        case LayerConnectionType.Straight:
                            newLayer = StraightConnectLayer<IHiddenNeuron>(prevLayer, identityGenerator, layerSettings, optimizers, () => neuronConstructor(layerSettings));
                            break;

                        default:
                            throw new ArgumentException("Invalid layer connection type");
                    }

                    if (layerSettings.CellType != NeuronType.Normal)
                    {
                        RecurrentConnectLayer<IHiddenNeuron>(newLayer, identityGenerator, layerSettings, optimizers);
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
                                                        IUniqueIdentityGenerator identityGenerator,
                                                        OutputLayerSettings settings,
                                                        bool shouldActivate,
                                                        ICollection<LearningRateOptimizer> optimizers)
        {
            return FullyConnectLayers<IOutputNeuron>(prevLayer, identityGenerator, settings, optimizers, () => new OutputNeuron(settings.ActivationType, shouldActivate));
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

        /// <param name="fanIn">Number of inputs of <paramref name="output"/>, for Xavier initialization; when 0 the
        /// neurons count of the layer is used</param>
        /// <param name="sharedOptimizer">Optimizer to use instead of a new one, already registered in
        /// <paramref name="optimizers"/>; see <see cref="SharedOptimizer"/></param>
        public IOptimizer ConnectAxon(INeuron input,
                                      INeuron output,
                                      IUniqueIdentityGenerator identityGenerator,
                                      CommonLayerSettings settings,
                                      ICollection<LearningRateOptimizer> optimizers,
                                      bool isConnectionActive = true,
                                      bool recurrentConnection = false,
                                      int fanIn = 0,
                                      IOptimizer sharedOptimizer = null)
        {
            Quad initialWeight;
            switch (settings.InitializationType)
            {
                case WeightInitType.Random:
                    initialWeight = NetworkUtils.RandomWeight();
                    break;

                case WeightInitType.Xavier:
                    initialWeight = NetworkUtils.GlorotWeight(fanIn > 0 ? fanIn : settings.NeuronsCount, settings.NeuronsCount);
                    break;

                default:
                    throw new ArgumentException("Invalid value", nameof(settings.InitializationType));
            }

            IOptimizer optimizer = sharedOptimizer;
            if (optimizer == null)
            {
                optimizer = NetworkUtils.GetOptimizer(settings.OptimizerType);
                if (optimizer is LearningRateOptimizer learningRateOptimizer)
                {
                    optimizers.Add(learningRateOptimizer);
                }
            }

            ISynapse axon;
            if (isConnectionActive)
            {
                axon = new ActiveSynapse(input, output, identityGenerator, optimizer, initialWeight);
            }
            else
            {
                axon = new InactiveSynapse(input, output, identityGenerator);
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

        /// <summary>
        /// 1-D convolution over the previous layer (without its bias neuron): output position o of filter f receives the
        /// inputs o * stride - padding + k for every kernel index k inside the signal, through the filter's shared kernel
        /// weights. Neurons are ordered filter by filter. A bias neuron of the previous layer feeds every neuron through
        /// one shared bias weight per filter
        /// </summary>
        private ILayer<T> ConvolutionalConnectLayers<T>(ILayer<INeuron> prevLayer,
                                                        IUniqueIdentityGenerator identityGenerator,
                                                        HiddenLayerSettings settings,
                                                        ICollection<LearningRateOptimizer> optimizers,
                                                        Func<T> neuronConstructor)
                       where T : class, INeuron
        {
            var (signal, bias) = SplitBias(prevLayer);
            int kernel = settings.KernelSize, stride = settings.Stride, padding = settings.Padding;
            ValidateKernel(settings);
            if (signal.Count + (2 * padding) < kernel)
                throw new ArgumentException($"A kernel of {kernel} does not fit {signal.Count} inputs with padding {padding}", nameof(settings));

            int positions = ((signal.Count + (2 * padding) - kernel) / stride) + 1;
            var layer = new Layer<T>(settings.FiltersCount * positions);

            for (int f = 0; f < settings.FiltersCount; f++)
            {
                SharedWeight[] weights = CreateKernel(kernel, kernel, settings, optimizers);
                SharedWeight biasWeight = bias == null ? null : CreateSharedWeight(0, settings, optimizers);

                for (int o = 0; o < positions; o++)
                {
                    T neuron = neuronConstructor();
                    layer.Add(neuron);

                    for (int k = 0; k < kernel; k++)
                    {
                        int input = (o * stride) - padding + k;
                        if (input >= 0 && input < signal.Count)
                        {
                            ConnectShared(signal[input], neuron, identityGenerator, weights[k]);
                        }
                    }

                    if (bias != null) ConnectShared(bias, neuron, identityGenerator, biasWeight);
                }
            }

            AddBiasNeuron(layer, settings);
            return layer;
        }

        /// <summary>
        /// 1-D transposed convolution: input i of the previous layer (without its bias neuron) adds to the outputs
        /// i * stride - padding + k of filter f through the filter's shared kernel weights. The output has
        /// (inputs - 1) * stride - 2 * padding + kernel positions per filter
        /// </summary>
        private ILayer<T> DeconvolutionalConnectLayers<T>(ILayer<INeuron> prevLayer,
                                                          IUniqueIdentityGenerator identityGenerator,
                                                          HiddenLayerSettings settings,
                                                          ICollection<LearningRateOptimizer> optimizers,
                                                          Func<T> neuronConstructor)
                       where T : class, INeuron
        {
            var (signal, bias) = SplitBias(prevLayer);
            int kernel = settings.KernelSize, stride = settings.Stride, padding = settings.Padding;
            ValidateKernel(settings);
            if (stride > kernel)
                throw new ArgumentException("The stride of a deconvolution must not exceed the kernel, or some outputs would get no input", nameof(settings));

            int positions = ((signal.Count - 1) * stride) - (2 * padding) + kernel;
            if (positions < 1)
                throw new ArgumentException($"The padding of {padding} removes the whole output", nameof(settings));

            var layer = new Layer<T>(settings.FiltersCount * positions);

            for (int f = 0; f < settings.FiltersCount; f++)
            {
                SharedWeight[] weights = CreateKernel(kernel, kernel, settings, optimizers);
                SharedWeight biasWeight = bias == null ? null : CreateSharedWeight(0, settings, optimizers);
                var outputs = new T[positions];

                for (int o = 0; o < positions; o++)
                {
                    outputs[o] = neuronConstructor();
                    layer.Add(outputs[o]);
                }

                for (int i = 0; i < signal.Count; i++)
                {
                    for (int k = 0; k < kernel; k++)
                    {
                        int output = (i * stride) - padding + k;
                        if (output >= 0 && output < positions)
                        {
                            ConnectShared(signal[i], outputs[output], identityGenerator, weights[k]);
                        }
                    }
                }

                if (bias != null)
                {
                    foreach (T output in outputs) ConnectShared(bias, output, identityGenerator, biasWeight);
                }
            }

            AddBiasNeuron(layer, settings);
            return layer;
        }

        /// <summary>
        /// One optimizer for all synapses of a layer when the optimizer keeps no state (plain gradient descent), so a
        /// layer holds one optimizer object instead of one per synapse; null for optimizers with state. Setting the
        /// learning rate through the optimizers collection works as before
        /// </summary>
        internal static IOptimizer SharedOptimizer(OptimizerTypes type, ICollection<LearningRateOptimizer> optimizers)
        {
            if (type != OptimizerTypes.Normal) return null;

            IOptimizer optimizer = NetworkUtils.GetOptimizer(type);
            optimizers.Add((LearningRateOptimizer)optimizer);
            return optimizer;
        }

        private static void ValidateKernel(HiddenLayerSettings settings)
        {
            if (settings.KernelSize < 1) throw new ArgumentException("The kernel must hold at least one weight", nameof(settings));
            if (settings.Stride < 1) throw new ArgumentException("The stride must be at least 1", nameof(settings));
            if (settings.FiltersCount < 1) throw new ArgumentException("At least one filter is required", nameof(settings));
            if (settings.Padding < 0 || settings.Padding >= settings.KernelSize)
                throw new ArgumentException("The padding must be at least 0 and smaller than the kernel, or some neurons would get no input", nameof(settings));
        }

        /// <summary>
        /// The signal neurons of a layer and its bias neuron, if it has one
        /// </summary>
        private static (List<INeuron> Signal, INeuron Bias) SplitBias(ILayer<INeuron> layer)
        {
            var signal = layer.Where(x => x is not BiasNeuron).ToList();
            if (signal.Count == 0) throw new ArgumentException("The previous layer has no neurons to convolve", nameof(layer));
            return (signal, layer.FirstOrDefault(x => x is BiasNeuron));
        }

        private static SharedWeight[] CreateKernel(int size, int fanIn, CommonLayerSettings settings, ICollection<LearningRateOptimizer> optimizers)
        {
            var kernel = new SharedWeight[size];
            for (int k = 0; k < size; k++)
            {
                Quad weight = settings.InitializationType == WeightInitType.Xavier ? NetworkUtils.GlorotWeight(fanIn, fanIn) : NetworkUtils.RandomWeight();
                kernel[k] = CreateSharedWeight(weight, settings, optimizers);
            }

            return kernel;
        }

        private static SharedWeight CreateSharedWeight(Quad value, CommonLayerSettings settings, ICollection<LearningRateOptimizer> optimizers)
        {
            IOptimizer optimizer = NetworkUtils.GetOptimizer(settings.OptimizerType);
            if (optimizer is LearningRateOptimizer learningRateOptimizer)
            {
                optimizers.Add(learningRateOptimizer);
            }

            return new SharedWeight(value, optimizer);
        }

        private static void ConnectShared(INeuron input, INeuron output, IUniqueIdentityGenerator identityGenerator, SharedWeight weight)
        {
            var synapse = new SharedWeightSynapse(input, output, identityGenerator, weight);
            output.Inputs.Add(synapse);
            input.Outputs.Add(synapse);
        }

        private static void AddBiasNeuron<T>(Layer<T> layer, CommonLayerSettings settings)
            where T : class, INeuron
        {
            if (settings.HasBiasNeuron && new BiasNeuron(settings.ActivationType) is T bias)
                layer.Add(bias);
        }

        private ILayer<T> StraightConnectLayer<T>(ILayer<INeuron> prevLayer,
                                                  IUniqueIdentityGenerator identityGenerator,
                                                  CommonLayerSettings settings,
                                                  ICollection<LearningRateOptimizer> optimizers,
                                                  Func<T> neuronConstructor)
               where T : class, INeuron
        {
            if (settings.NeuronsCount != prevLayer.Count)
                throw new ArgumentException("Бройката неврони в двата слоя трябва да е еднаква");

            var layer = new Layer<T>(prevLayer.Count + 1);

            IOptimizer shared = SharedOptimizer(settings.OptimizerType, optimizers);
            for (int i = 0; i < prevLayer.Count; i++)
            {
                var neuron = neuronConstructor();
                layer.Add(neuron);
                ConnectAxon(prevLayer[i], neuron, identityGenerator, settings, optimizers, sharedOptimizer: shared);
            }

            return layer;
        }

        private ILayer<T> FullyConnectLayers<T>(IEnumerable<INeuron> prevLayer, IUniqueIdentityGenerator identityGenerator, CommonLayerSettings settings, ICollection<LearningRateOptimizer> optimizers, Func<T> neuronConstructor)
             where T : class, INeuron
        {
            int fanIn = prevLayer.Count();
            IOptimizer shared = SharedOptimizer(settings.OptimizerType, optimizers);
            var layer = new Layer<T>(settings.NeuronsCount);

            for (int i = 0; i < settings.NeuronsCount; i++)
            {
                var neuron = neuronConstructor();
                layer.Add(neuron);
                foreach (var inNeuron in prevLayer)
                {
                    ConnectAxon(inNeuron, neuron, identityGenerator, settings, optimizers, fanIn: fanIn, sharedOptimizer: shared);
                }
            }

            if (settings.HasBiasNeuron && new BiasNeuron(settings.ActivationType) is T bias)
                layer.Add(bias);

            return layer;
        }

        private void RecurrentConnectLayer<T>(ILayer<T> layer, IUniqueIdentityGenerator identityGenerator, RecurrentLayerSettings settings, ICollection<LearningRateOptimizer> optimizers)
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
                            ConnectAxon(layer[i], layer[j], identityGenerator, settings, optimizers, true);
                        }
                    }
                    else
                    {
                        ConnectAxon(layer[i], layer[i + 1], identityGenerator, settings, optimizers, true);
                    }

                    if (settings.HasSelfConnection)
                    {
                        ConnectAxon(layer[i], layer[i], identityGenerator, settings, optimizers, true);
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
                            ConnectAxon(layer[i], layer[j], identityGenerator, settings, optimizers, true);
                        }
                    }
                    else
                    {
                        ConnectAxon(layer[i], layer[i - 1], identityGenerator, settings, optimizers, true);
                    }

                    if (settings.HasSelfConnection && (settings.Direction & RecurrentDirection.Forward) != RecurrentDirection.Forward)
                    {
                        ConnectAxon(layer[i], layer[i], identityGenerator, settings, optimizers, true);
                    }
                }
            }
        }
    }
}
