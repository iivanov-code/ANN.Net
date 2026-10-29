using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Settings;

namespace ANN.Net.Remote
{
    /// <summary>
    /// Builds the pieces of a distributed network from one description. Every process describes the whole
    /// network the same way. A depth cut after a hidden layer gives a head (<see cref="BuildHead(NetworkSettings, RecurrentLayerSettings[], int, SynapseConnection)"/>)
    /// and a tail (<see cref="BuildTail(NetworkSettings, RecurrentLayerSettings[], int)"/>, served with <see cref="RemoteNetworkHost"/>);
    /// a breadth cut of a run of layers (<see cref="Partition(NetworkSettings, RecurrentLayerSettings[], int, int, int)"/>) gives one
    /// <see cref="PartNetwork"/> per part, run by <see cref="PartitionNode"/>s and fronted by a <see cref="PartitionCoordinator"/>
    /// </summary>
    public static class RemoteNetworkFactory
    {
        /// <summary>
        /// Cuts the description after hidden layer <paramref name="boundaryLayer"/> (0-based, counting every layer
        /// a <see cref="HiddenLayerSettings.LayersCount"/> expands to). The boundary layer's neurons stay in the
        /// head, their outgoing synapses and bias move to the tail
        /// </summary>
        public static NetworkSplit Split(NetworkSettings settings, RecurrentLayerSettings[] hiddenLayers, int boundaryLayer)
        {
            if (settings == null) throw new ArgumentNullException(nameof(settings));
            if (settings.InputSettings == null) throw new ArgumentException("Input settings are required", nameof(settings));
            if (settings.OutputSettings == null) throw new ArgumentException("Output settings are required", nameof(settings));
            if (settings.IdentityGenerator == null) throw new ArgumentException("An identity generator is required; the remote synapses are matched by id", nameof(settings));

            RecurrentLayerSettings[] layers = Expand(hiddenLayers);
            if (layers.Length == 0) throw new ArgumentException("A network needs at least one hidden layer to be split", nameof(hiddenLayers));
            if (boundaryLayer < 0 || boundaryLayer >= layers.Length)
                throw new ArgumentOutOfRangeException(nameof(boundaryLayer), $"Must be between 0 and {layers.Length - 1}");

            RecurrentLayerSettings boundary = layers[boundaryLayer];
            if (boundary.CellType != NeuronType.Normal)
                throw new NotSupportedException("Only a layer of normal neurons can be the boundary");
            if (boundary.PrevLayerConnectionType != LayerConnectionType.FullyConnected)
                throw new NotSupportedException("The boundary layer must be fully connected to the layer before it");
            if (boundary.NeuronsCount < 1)
                throw new ArgumentException("The boundary layer has no neurons", nameof(hiddenLayers));

            NetworkSettings head = new NetworkSettings
            {
                IdentityGenerator = settings.IdentityGenerator,
                LossType = LossFunctionTypes.Squared,
                OutputApplySoftMax = false,
                ShouldNormalize = settings.ShouldNormalize,
                MinMaxValues = settings.MinMaxValues,
                InputSettings = settings.InputSettings,
                OutputSettings = new OutputLayerSettings
                {
                    NeuronsCount = boundary.NeuronsCount,
                    ActivationType = boundary.ActivationType,
                    InitializationType = boundary.InitializationType,
                    OptimizerType = boundary.OptimizerType,
                    HasBiasNeuron = false
                }
            };

            NetworkSettings tail = new NetworkSettings
            {
                IdentityGenerator = settings.IdentityGenerator,
                LossType = settings.LossType,
                OutputApplySoftMax = settings.OutputApplySoftMax,
                ShouldNormalize = false,
                MinMaxValues = null,
                InputSettings = new InputLayerSettings
                {
                    NeuronsCount = boundary.NeuronsCount,
                    ActivationType = ActivationTypes.Identity,
                    HasBiasNeuron = boundary.HasBiasNeuron,
                    InitializationType = boundary.InitializationType,
                    OptimizerType = boundary.OptimizerType
                },
                OutputSettings = settings.OutputSettings
            };

            return new NetworkSplit
            {
                HeadSettings = head,
                HeadHiddenLayers = Slice(layers, 0, boundaryLayer),
                TailSettings = tail,
                TailHiddenLayers = Slice(layers, boundaryLayer + 1, layers.Length - boundaryLayer - 1),
                BoundaryLayer = boundaryLayer,
                BoundaryNeurons = boundary.NeuronsCount
            };
        }

        public static NetworkSplit Split(NetworkSettings settings, RecurrentLayerSettings hiddenLayers, int boundaryLayer)
        {
            return Split(settings, hiddenLayers == null ? null : new[] { hiddenLayers }, boundaryLayer);
        }

        /// <summary>
        /// Builds the local half and attaches one remote synapse per boundary neuron to <paramref name="connection"/>
        /// </summary>
        public static SplitNetwork BuildHead(NetworkSettings settings, RecurrentLayerSettings[] hiddenLayers, int boundaryLayer, SynapseConnection connection)
        {
            return BuildHead(Split(settings, hiddenLayers, boundaryLayer), connection);
        }

        public static SplitNetwork BuildHead(NetworkSettings settings, RecurrentLayerSettings hiddenLayers, int boundaryLayer, SynapseConnection connection)
        {
            return BuildHead(Split(settings, hiddenLayers, boundaryLayer), connection);
        }

        public static SplitNetwork BuildHead(NetworkSplit split, SynapseConnection connection)
        {
            if (split == null) throw new ArgumentNullException(nameof(split));
            if (connection == null) throw new ArgumentNullException(nameof(connection));

            Network head = (Network)NetworkFactory.BuildFFN(split.HeadSettings, null, split.HeadHiddenLayers);
            return new SplitNetwork(head, connection, split.HeadSettings.IdentityGenerator);
        }

        /// <summary>
        /// Builds the remote half: an ordinary network whose input layer receives the boundary values
        /// </summary>
        public static INetwork BuildTail(NetworkSettings settings, RecurrentLayerSettings[] hiddenLayers, int boundaryLayer)
        {
            return BuildTail(Split(settings, hiddenLayers, boundaryLayer));
        }

        public static INetwork BuildTail(NetworkSettings settings, RecurrentLayerSettings hiddenLayers, int boundaryLayer)
        {
            return BuildTail(Split(settings, hiddenLayers, boundaryLayer));
        }

        public static INetwork BuildTail(NetworkSplit split)
        {
            if (split == null) throw new ArgumentNullException(nameof(split));

            return NetworkFactory.BuildFFN(split.TailSettings, null, split.TailHiddenLayers);
        }

        /// <summary>
        /// Describes hidden layers <paramref name="firstLayer"/> to <paramref name="lastLayer"/> (0-based, every
        /// layer a <see cref="HiddenLayerSettings.LayersCount"/> expands to counts) cut in breadth into
        /// <paramref name="parts"/> slices. The section sits behind a head cut after <paramref name="firstLayer"/> - 1
        /// and in front of a tail cut after <paramref name="lastLayer"/>; the bias of the layer before the section
        /// is repeated in every part, the bias of the last section layer belongs to the tail's input layer
        /// </summary>
        public static NetworkPartition Partition(NetworkSettings settings, RecurrentLayerSettings[] hiddenLayers, int firstLayer, int lastLayer, int parts)
        {
            if (settings == null) throw new ArgumentNullException(nameof(settings));
            if (settings.InputSettings == null) throw new ArgumentException("Input settings are required", nameof(settings));
            if (settings.IdentityGenerator == null) throw new ArgumentException("An identity generator is required; the remote synapses are matched by id", nameof(settings));
            if (parts < 1) throw new ArgumentOutOfRangeException(nameof(parts), "Must be greater than 0");

            RecurrentLayerSettings[] layers = Expand(hiddenLayers);
            if (firstLayer < 0 || firstLayer >= layers.Length) throw new ArgumentOutOfRangeException(nameof(firstLayer), $"Must be between 0 and {layers.Length - 1}");
            if (lastLayer < firstLayer || lastLayer >= layers.Length) throw new ArgumentOutOfRangeException(nameof(lastLayer), $"Must be between {firstLayer} and {layers.Length - 1}");

            RecurrentLayerSettings[] section = Slice(layers, firstLayer, lastLayer - firstLayer + 1);
            foreach (RecurrentLayerSettings layer in section)
            {
                if (layer.CellType != NeuronType.Normal) throw new NotSupportedException("Only layers of normal neurons can be cut in breadth");
                if (layer.PrevLayerConnectionType != LayerConnectionType.FullyConnected) throw new NotSupportedException("Only fully connected layers can be cut in breadth");
                if (layer.NeuronsCount < parts) throw new ArgumentException($"A layer of {layer.NeuronsCount} neurons cannot be cut into {parts} parts", nameof(parts));
            }

            int inputWidth = firstLayer == 0 ? settings.InputSettings.NeuronsCount : layers[firstLayer - 1].NeuronsCount;
            bool inputHasBias = firstLayer == 0 ? settings.InputSettings.HasBiasNeuron : layers[firstLayer - 1].HasBiasNeuron;
            if (inputWidth < 1) throw new ArgumentException("The layer feeding the section has no neurons", nameof(hiddenLayers));

            return new NetworkPartition(settings.IdentityGenerator, section, firstLayer, inputWidth, inputHasBias, parts);
        }

        public static NetworkPartition Partition(NetworkSettings settings, RecurrentLayerSettings hiddenLayers, int firstLayer, int lastLayer, int parts)
        {
            return Partition(settings, hiddenLayers == null ? null : new[] { hiddenLayers }, firstLayer, lastLayer, parts);
        }

        /// <summary>
        /// Builds one part's slice of a partitioned section
        /// </summary>
        public static PartNetwork BuildPart(NetworkPartition partition, int part)
        {
            return new PartNetwork(partition, part);
        }

        /// <summary>
        /// One entry per actual layer, each with LayersCount 1
        /// </summary>
        private static RecurrentLayerSettings[] Expand(RecurrentLayerSettings[] hiddenLayers)
        {
            List<RecurrentLayerSettings> layers = new List<RecurrentLayerSettings>();
            if (hiddenLayers == null) return layers.ToArray();

            foreach (RecurrentLayerSettings settings in hiddenLayers)
            {
                if (settings == null) continue;

                for (int i = 0; i < settings.LayersCount; i++)
                {
                    layers.Add(Clone(settings));
                }
            }

            return layers.ToArray();
        }

        private static RecurrentLayerSettings Clone(RecurrentLayerSettings settings)
        {
            return new RecurrentLayerSettings
            {
                LayersCount = 1,
                PrevLayerConnectionType = settings.PrevLayerConnectionType,
                CellType = settings.CellType,
                HasSelfConnection = settings.HasSelfConnection,
                FullyConnectLayer = settings.FullyConnectLayer,
                ActiveConnections = settings.ActiveConnections,
                Direction = settings.Direction,
                WindowSize = settings.WindowSize,
                HasBiasNeuron = settings.HasBiasNeuron,
                InitializationType = settings.InitializationType,
                OptimizerType = settings.OptimizerType,
                NeuronsCount = settings.NeuronsCount,
                ActivationType = settings.ActivationType
            };
        }

        private static RecurrentLayerSettings[] Slice(RecurrentLayerSettings[] layers, int start, int count)
        {
            RecurrentLayerSettings[] slice = new RecurrentLayerSettings[count];
            Array.Copy(layers, start, slice, 0, count);
            return slice;
        }
    }
}
