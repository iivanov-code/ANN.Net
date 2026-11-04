using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using ANN.Net.Abstractions.Collections;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Abstractions.Interfaces.Synapses;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Connections;
using ANN.Net.IdentityGenerators;
using ANN.Net.Neurons;
using ANN.Net.Serialization;
using ANN.Net.Utils;

namespace ANN.Net
{
    /// <summary>
    /// Saves networks to JSON and loads them back. Supports the networks of <see cref="NetworkFactory.BuildFFN(NetworkSettings, RecurrentLayerSettings)"/>,
    /// <see cref="NetworkFactory.BuildGraph"/> and <see cref="NetworkFactory.BuildConvolutional"/>: topology, weights, activations, optimizer types and learning rates,
    /// loss, normalization and soft-max. The internal state of the optimizers (Adam moments and the like) is not saved,
    /// so training continues as with freshly created optimizers
    /// </summary>
    public static class NetworkSerializer
    {
        public const string FileExtension = "ann";

        private const string FormatName = "ANN.Net network";
        private const int FormatVersion = 1;

        private static readonly JsonSerializerOptions JsonOptions = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
            Converters = { new JsonStringEnumConverter() }
        };

        public static void Save(INetwork network, string filePath)
        {
            ArgumentException.ThrowIfNullOrEmpty(filePath);

            using FileStream stream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
            Save(network, stream);
        }

        public static void Save(INetwork network, Stream stream)
        {
            ArgumentNullException.ThrowIfNull(network);
            ArgumentNullException.ThrowIfNull(stream);

            JsonSerializer.Serialize(stream, ToDocument(network), JsonOptions);
        }

        public static INetwork Load(string filePath)
        {
            ArgumentException.ThrowIfNullOrEmpty(filePath);

            using FileStream stream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            return Load(stream);
        }

        public static INetwork Load(Stream stream)
        {
            ArgumentNullException.ThrowIfNull(stream);

            NetworkDocument document;
            try
            {
                document = JsonSerializer.Deserialize<NetworkDocument>(stream, JsonOptions);
            }
            catch (JsonException e)
            {
                throw new InvalidDataException("The file is not a valid ANN.Net network", e);
            }

            try
            {
                return FromDocument(document);
            }
            catch (ArgumentException e)
            {
                // unknown activation, optimizer or loss types
                throw new InvalidDataException("The file holds settings this version of ANN.Net does not support", e);
            }
        }

        /// <summary>
        /// Loads a network of a known type, for example <see cref="IGraphNetwork"/>
        /// </summary>
        public static TNetwork Load<TNetwork>(string filePath)
            where TNetwork : INetwork
        {
            return Cast<TNetwork>(Load(filePath));
        }

        /// <summary>
        /// Loads a network of a known type, for example <see cref="IGraphNetwork"/>
        /// </summary>
        public static TNetwork Load<TNetwork>(Stream stream)
            where TNetwork : INetwork
        {
            return Cast<TNetwork>(Load(stream));
        }

        private static TNetwork Cast<TNetwork>(INetwork network)
            where TNetwork : INetwork
        {
            if (network is TNetwork result) return result;

            throw new InvalidDataException($"The file holds a {network.GetType().Name}, which is not a {typeof(TNetwork).Name}");
        }

        internal static NetworkDocument ToDocument(INetwork network)
        {
            Network layered;
            NetworkKind kind;
            Dictionary<BaseSynapse, uint> connectionIds = null;

            switch (network)
            {
                case Convolutional.ConvolutionalNetwork convolutional:
                    return new NetworkDocument
                    {
                        Format = FormatName,
                        Version = FormatVersion,
                        Kind = NetworkKind.Convolutional,
                        Convolutional = ConvolutionalSerializer.ToDocument(convolutional)
                    };

                case Matrix.MatrixNetwork matrix:
                    return MatrixSerializer.ToDocument(matrix, FormatName, FormatVersion);

                case Network n:
                    layered = n;
                    kind = NetworkKind.Layered;
                    break;

                case GraphNetwork graph:
                    layered = graph.Network;
                    kind = NetworkKind.Graph;
                    connectionIds = graph.Synapses.ToDictionary(x => (BaseSynapse)x.Value, x => x.Key);
                    break;

                default:
                    throw new NotSupportedException($"Saving a {network.GetType().Name} is not supported");
            }

            var document = new NetworkDocument
            {
                Format = FormatName,
                Version = FormatVersion,
                Kind = kind,
                LossType = layered.LossType,
                ShouldNormalize = layered.ShouldNormalize,
                OutputApplySoftMax = layered.ApplySoftMax,
                MinMaxValues = layered.MinMaxValues?.Select(x => new[] { (float)x.Item1, (float)x.Item2 }).ToList()
            };

            // neurons in layer order, remembering the layer of each one to recognize recurrent synapses
            var neurons = new List<INeuron>();
            var indexes = new Dictionary<INeuron, int>(ReferenceEqualityComparer.Instance);
            var layerOf = new Dictionary<INeuron, int>(ReferenceEqualityComparer.Instance);

            List<int> AddLayer(IEnumerable<INeuron> layer, int layerIndex)
            {
                var result = new List<int>();
                foreach (INeuron neuron in layer ?? Enumerable.Empty<INeuron>())
                {
                    if (!indexes.TryAdd(neuron, neurons.Count))
                        throw new NotSupportedException($"Neuron {neuron.ID} appears in more than one layer");

                    layerOf.Add(neuron, layerIndex);
                    result.Add(neurons.Count);
                    neurons.Add(neuron);
                }

                return result;
            }

            document.InputLayer = AddLayer(layered.InputNeurons, 0);
            int hiddenLayersCount = 0;
            foreach (ILayer<IHiddenNeuron> layer in layered.HiddenLayers ?? Enumerable.Empty<ILayer<IHiddenNeuron>>())
            {
                document.HiddenLayers.Add(AddLayer(layer, ++hiddenLayersCount));
            }

            document.OutputLayer = AddLayer(layered.OutputNeurons, hiddenLayersCount + 1);

            var synapseIndexes = new Dictionary<BaseSynapse, int>(ReferenceEqualityComparer.Instance);
            var sharedIndexes = new Dictionary<SharedWeight, int>(ReferenceEqualityComparer.Instance);

            foreach (INeuron neuron in neurons)
            {
                document.Neurons.Add(ToDocument(neuron));

                foreach (ISynapse synapse in neuron.Outputs ?? Enumerable.Empty<ISynapse>())
                {
                    BaseSynapse active = AsTrainableSynapse(synapse);
                    if (!indexes.TryGetValue(active.Output, out int to))
                        throw new NotSupportedException($"Neuron {neuron.ID} is connected to neuron {active.Output.ID}, which is not part of the network");

                    var synapseDocument = new SynapseDocument
                    {
                        Id = connectionIds?[active],
                        From = indexes[neuron],
                        To = to,
                        Weight = active.Weight,
                        Recurrent = kind == NetworkKind.Layered && layerOf[neuron] >= layerOf[active.Output],
                        Optimizer = active.Optimizer.GetOptimizerType(),
                        LearningRate = active.Optimizer is LearningRateOptimizer optimizer ? optimizer.LearningRate : null
                    };

                    if (active is SharedWeightSynapse shared)
                    {
                        synapseDocument.SharedWeight = SharedIndex(shared.SharedWeight, document, sharedIndexes);
                        synapseDocument.LearningRate = null;
                    }

                    synapseIndexes.Add(active, document.Synapses.Count);
                    document.Synapses.Add(synapseDocument);
                }
            }

            for (int i = 0; i < neurons.Count; i++)
            {
                if (neurons[i].Inputs == null) continue;

                document.Neurons[i].Inputs = neurons[i].Inputs
                    .Select(x => synapseIndexes.TryGetValue(AsTrainableSynapse(x), out int index)
                        ? index
                        : throw new NotSupportedException($"Neuron {neurons[i].ID} has an input synapse from a neuron outside the network"))
                    .ToList();
            }

            return document;
        }

        private static NeuronDocument ToDocument(INeuron neuron)
        {
            Type type = neuron.GetType();
            if (type == typeof(BiasNeuron))
                return new NeuronDocument { Id = neuron.ID, Type = NodeType.BIAS, Activation = ActivationTypes.Identity };

            NodeType nodeType;
            if (type == typeof(InputNeuron)) nodeType = NodeType.INPUT;
            else if (type == typeof(HiddenNeuron)) nodeType = NodeType.HIDDEN;
            else if (type == typeof(OutputNeuron)) nodeType = NodeType.OUTPUT;
            else throw new NotSupportedException($"Saving a {type.Name} is not supported");

            return new NeuronDocument
            {
                Id = neuron.ID,
                Type = nodeType,
                Activation = ((Neuron)neuron).GetActivationType(),
                ApplyActivation = neuron is OutputNeuron output ? output.ApplyActivation : null
            };
        }

        private static int SharedIndex(SharedWeight weight, NetworkDocument document, Dictionary<SharedWeight, int> indexes)
        {
            if (!indexes.TryGetValue(weight, out int index))
            {
                document.SharedWeights ??= new List<SharedWeightDocument>();
                index = document.SharedWeights.Count;
                indexes.Add(weight, index);
                document.SharedWeights.Add(new SharedWeightDocument
                {
                    Weight = weight.Value,
                    Optimizer = weight.Optimizer.GetOptimizerType(),
                    LearningRate = weight.Optimizer is LearningRateOptimizer optimizer ? optimizer.LearningRate : null
                });
            }

            return index;
        }

        /// <summary>
        /// Synapses with an own or a shared weight; other synapse types (sequence, remote) cannot be saved
        /// </summary>
        private static BaseSynapse AsTrainableSynapse(ISynapse synapse)
        {
            if (synapse != null && (synapse.GetType() == typeof(ActiveSynapse) || synapse is SharedWeightSynapse)) return (BaseSynapse)synapse;

            throw new NotSupportedException($"Saving a {synapse?.GetType().Name ?? "missing synapse"} is not supported");
        }

        private static INetwork FromDocument(NetworkDocument document)
        {
            if (document == null || !string.Equals(document.Format, FormatName, StringComparison.Ordinal))
                throw new InvalidDataException("The file is not an ANN.Net network");
            if (document.Version != FormatVersion)
                throw new InvalidDataException($"Unsupported network file version {document.Version}");

            if (document.Kind == NetworkKind.Convolutional)
                return ConvolutionalSerializer.FromDocument(document.Convolutional);

            List<NeuronDocument> neuronDocuments = document.Neurons ?? new List<NeuronDocument>();
            List<SynapseDocument> synapseDocuments = document.Synapses ?? new List<SynapseDocument>();

            INeuron[] neurons = neuronDocuments.Select(CreateNeuron).ToArray();

            ILayer<T> Layer<T>(List<int> indexes, string name)
                where T : class, INeuron
            {
                var layer = new Layer<T>();
                foreach (int index in indexes ?? new List<int>())
                {
                    if (NeuronAt(index) is not T neuron)
                        throw new InvalidDataException($"Neuron {index} cannot be part of the {name} layer");

                    layer.Add(neuron);
                }

                return layer;
            }

            INeuron NeuronAt(int index)
            {
                if (index < 0 || index >= neurons.Length)
                    throw new InvalidDataException($"Unknown neuron index {index}");

                return neurons[index];
            }

            ILayer<IInputNeuron> inputLayer = Layer<IInputNeuron>(document.InputLayer, "input");
            List<ILayer<IHiddenNeuron>> hiddenLayers = (document.HiddenLayers ?? new List<List<int>>()).Select(x => Layer<IHiddenNeuron>(x, "hidden")).ToList();
            ILayer<IOutputNeuron> outputLayer = Layer<IOutputNeuron>(document.OutputLayer, "output");

            var network = new Network(new NetworkSettings
            {
                LossType = document.LossType,
                ShouldNormalize = document.ShouldNormalize,
                OutputApplySoftMax = document.OutputApplySoftMax,
                MinMaxValues = document.MinMaxValues?.Select(ToMinMax).ToList(),
                InputSettings = new InputLayerSettings { NeuronsCount = (ushort)inputLayer.Count(x => x is not BiasNeuron) },
                OutputSettings = new OutputLayerSettings { NeuronsCount = (ushort)outputLayer.Count }
            })
            {
                InputNeurons = inputLayer,
                HiddenLayers = hiddenLayers,
                OutputNeurons = outputLayer
            };

            var ids = new IncrementalGenerator();
            var synapses = new ISynapse[synapseDocuments.Count];
            var sharedOptimizers = new Dictionary<float, IOptimizer>();
            SharedWeight[] sharedWeights = (document.SharedWeights ?? new List<SharedWeightDocument>())
                .Select(x => x == null
                    ? throw new InvalidDataException("A shared weight is missing")
                    : new SharedWeight(x.Weight, CreateOptimizer(x.Optimizer, x.LearningRate, network.Optimizers)))
                .ToArray();

            for (int i = 0; i < synapses.Length; i++)
            {
                SynapseDocument synapse = synapseDocuments[i] ?? throw new InvalidDataException($"Synapse {i} is missing");
                INeuron from = NeuronAt(synapse.From);
                INeuron to = NeuronAt(synapse.To);
                if (from.Outputs == null || to.Inputs == null)
                    throw new InvalidDataException($"Synapse {i} connects neurons that cannot be connected that way");

                if (synapse.SharedWeight is int shared)
                {
                    if (shared < 0 || shared >= sharedWeights.Length)
                        throw new InvalidDataException($"Synapse {i} refers to an unknown shared weight {shared}");

                    synapses[i] = new SharedWeightSynapse(from, to, ids, sharedWeights[shared]);
                }
                else
                {
                    synapses[i] = new ActiveSynapse(from, to, ids, OptimizerFor(synapse, network.Optimizers, sharedOptimizers), synapse.Weight);
                }

                if (synapse.Recurrent) from.Outputs.AddRecurrent(synapses[i]);
                else from.Outputs.Add(synapses[i]);
            }

            // input synapses in the order every neuron held them, so sums and gradients add up in the same order
            var connected = new bool[synapses.Length];
            for (int i = 0; i < neurons.Length; i++)
            {
                foreach (int index in neuronDocuments[i].Inputs ?? new List<int>())
                {
                    if (index < 0 || index >= synapses.Length || synapseDocuments[index].To != i || connected[index])
                        throw new InvalidDataException($"Neuron {i} lists an invalid input synapse {index}");

                    connected[index] = true;
                    if (synapseDocuments[index].Recurrent) neurons[i].Inputs.AddRecurrent(synapses[index]);
                    else neurons[i].Inputs.Add(synapses[index]);
                }
            }

            int missing = Array.IndexOf(connected, false);
            if (missing >= 0)
                throw new InvalidDataException($"Synapse {missing} is not listed among the inputs of its neuron");

            switch (document.Kind)
            {
                case NetworkKind.Layered:
                    return document.Matrix == true ? Matrix.MatrixNetwork.FromNetwork(network) : network;

                case NetworkKind.Graph:
                    var connections = new Dictionary<uint, ActiveSynapse>();
                    for (int i = 0; i < synapses.Length; i++)
                    {
                        uint id = synapseDocuments[i].Id ?? throw new InvalidDataException($"Synapse {i} of a graph network has no id");
                        if (synapses[i] is not ActiveSynapse own)
                            throw new InvalidDataException($"Synapse {i} of a graph network shares its weight");
                        if (!connections.TryAdd(id, own))
                            throw new InvalidDataException($"Duplicate connection id {id}");
                    }

                    List<uint> hiddenIds = hiddenLayers.SelectMany(x => x).Where(x => x is HiddenNeuron).Select(x => x.ID).ToList();
                    return GraphNetwork.FromNetwork(network, connections, hiddenIds);

                default:
                    throw new InvalidDataException($"Unknown network kind {document.Kind}");
            }
        }

        private static INeuron CreateNeuron(NeuronDocument neuron, int index)
        {
            if (neuron == null) throw new InvalidDataException($"Neuron {index} is missing");

            switch (neuron.Type)
            {
                case NodeType.INPUT:
                    return new InputNeuron(neuron.Activation, id: neuron.Id);
                case NodeType.HIDDEN:
                    return new HiddenNeuron(neuron.Activation, neuron.Id);
                case NodeType.OUTPUT:
                    return new OutputNeuron(neuron.Activation, neuron.ApplyActivation ?? true, neuron.Id);
                case NodeType.BIAS:
                    return new BiasNeuron(neuron.Activation, neuron.Id);
                default:
                    throw new InvalidDataException($"Unknown neuron type {neuron.Type}");
            }
        }

        /// <summary>
        /// Plain gradient descent keeps no state, so synapses with the same learning rate share one optimizer
        /// </summary>
        private static IOptimizer OptimizerFor(SynapseDocument synapse, ICollection<LearningRateOptimizer> optimizers, Dictionary<float, IOptimizer> shared)
        {
            if (synapse.Optimizer != OptimizerTypes.Normal) return CreateOptimizer(synapse.Optimizer, synapse.LearningRate, optimizers);

            float rate = synapse.LearningRate ?? 0;
            if (!shared.TryGetValue(rate, out IOptimizer optimizer))
            {
                optimizer = CreateOptimizer(OptimizerTypes.Normal, synapse.LearningRate, optimizers);
                shared.Add(rate, optimizer);
            }

            return optimizer;
        }

        private static IOptimizer CreateOptimizer(OptimizerTypes type, float? learningRate, ICollection<LearningRateOptimizer> optimizers)
        {
            IOptimizer optimizer = NetworkUtils.GetOptimizer(type);
            if (optimizer is LearningRateOptimizer learningRateOptimizer)
            {
                if (learningRate.HasValue)
                {
                    if (learningRate < 0 || learningRate > 1)
                        throw new InvalidDataException($"Invalid learning rate {learningRate}");

                    learningRateOptimizer.LearningRate = learningRate.Value;
                }

                optimizers.Add(learningRateOptimizer);
            }

            return optimizer;
        }

        private static Tuple<Quad, Quad> ToMinMax(float[] values)
        {
            if (values == null || values.Length != 2)
                throw new InvalidDataException("Every min/max entry must hold exactly two values");

            return Tuple.Create<Quad, Quad>(values[0], values[1]);
        }
    }
}
