using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Nested;

namespace ANN.Net.Serialization
{
    internal sealed class NestedDocument
    {
        public int InputSize { get; set; }

        public LossFunctionTypes LossType { get; set; }

        public List<NestedNodeDocument> Nodes { get; set; } = new List<NestedNodeDocument>();

        public List<uint> Outputs { get; set; } = new List<uint>();
    }

    internal sealed class NestedNodeDocument
    {
        public uint Id { get; set; }

        public List<uint> Inputs { get; set; }

        public NodeCombination Combination { get; set; }

        /// <summary>
        /// The network inside the node, as it would be saved on its own
        /// </summary>
        public NetworkDocument Network { get; set; }
    }

    /// <summary>
    /// Nested networks keep the document of the network of every node, so they nest in the file as they do in memory
    /// </summary>
    internal static class NestedSerializer
    {
        public static NestedDocument ToDocument(NestedNetwork network)
        {
            return new NestedDocument
            {
                InputSize = network.InputSize,
                LossType = network.LossType,
                Outputs = network.OutputIds.ToList(),
                Nodes = network.NodeList.Select(node => new NestedNodeDocument
                {
                    Id = node.Id,
                    Inputs = node.Inputs.ToList(),
                    Combination = node.Combination,
                    Network = node.Network is INetwork inner
                        ? NetworkSerializer.ToDocument(inner)
                        : throw new NotSupportedException($"Node {node.Id} holds a {node.Network.GetType().Name}, which cannot be saved (transformers cannot be saved yet)")
                }).ToList()
            };
        }

        public static NestedNetwork FromDocument(NestedDocument document, Func<NetworkDocument, INetwork> load)
        {
            if (document?.Nodes == null || document.Outputs == null) throw new InvalidDataException("The nested network has no nodes");

            var settings = new NestedNetworkSettings { InputSize = document.InputSize, LossType = document.LossType, Outputs = document.Outputs };
            foreach (NestedNodeDocument node in document.Nodes)
            {
                if (node?.Network == null) throw new InvalidDataException("A node of the nested network has no network");
                if (load(node.Network) is not IVectorNetwork network) throw new InvalidDataException($"The network of node {node.Id} cannot be inside a nested network");

                settings.Nodes.Add(new NestedNode { Id = node.Id, Network = network, Inputs = node.Inputs ?? new List<uint>(), Combination = node.Combination });
            }

            try
            {
                return NestedNetwork.Build(settings);
            }
            catch (ArgumentException exception)
            {
                throw new InvalidDataException($"Invalid nested network: {exception.Message}", exception);
            }
        }
    }
}
