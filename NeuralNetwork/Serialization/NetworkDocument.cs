using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;

namespace ANN.Net.Serialization
{
    /// <summary>
    /// File contents of a saved network. Neurons and synapses refer to each other by their index in
    /// <see cref="Neurons"/> and <see cref="Synapses"/>, so the file does not depend on neuron ids being unique
    /// </summary>
    internal sealed class NetworkDocument
    {
        public string Format { get; set; }

        public int Version { get; set; }

        public NetworkKind Kind { get; set; }

        public LossFunctionTypes LossType { get; set; }

        public bool ShouldNormalize { get; set; }

        /// <summary>
        /// [min, max] of every input/output value, used when <see cref="ShouldNormalize"/> is set
        /// </summary>
        public List<float[]> MinMaxValues { get; set; }

        public bool OutputApplySoftMax { get; set; }

        public List<int> InputLayer { get; set; } = new List<int>();

        public List<List<int>> HiddenLayers { get; set; } = new List<List<int>>();

        public List<int> OutputLayer { get; set; } = new List<int>();

        public List<NeuronDocument> Neurons { get; set; } = new List<NeuronDocument>();

        /// <summary>
        /// Every synapse, in the order of the output synapses of the neurons
        /// </summary>
        public List<SynapseDocument> Synapses { get; set; } = new List<SynapseDocument>();

        public List<SharedWeightDocument> SharedWeights { get; set; }

        /// <summary>
        /// Layers of a <see cref="NetworkKind.Convolutional"/> network
        /// </summary>
        public ConvolutionalDocument Convolutional { get; set; }
    }

    internal enum NetworkKind
    {
        /// <summary>
        /// Layered network from <c>NetworkFactory.BuildFFN</c>
        /// </summary>
        Layered = 0,

        /// <summary>
        /// Free-form network from <c>NetworkFactory.BuildGraph</c>
        /// </summary>
        Graph = 1,

        /// <summary>
        /// Convolutional network from <c>NetworkFactory.BuildConvolutional</c>; stored in <see cref="NetworkDocument.Convolutional"/>
        /// </summary>
        Convolutional = 2
    }

    internal sealed class NeuronDocument
    {
        public uint Id { get; set; }

        public NodeType Type { get; set; }

        public ActivationTypes Activation { get; set; }

        /// <summary>
        /// Output neurons only: whether the activation is applied (it is not when the network applies soft-max)
        /// </summary>
        public bool? ApplyActivation { get; set; }

        /// <summary>
        /// Indexes of the input synapses in the order the neuron holds them
        /// </summary>
        public List<int> Inputs { get; set; }
    }

    internal sealed class SynapseDocument
    {
        /// <summary>
        /// Connection id of a graph network
        /// </summary>
        public uint? Id { get; set; }

        public int From { get; set; }

        public int To { get; set; }

        public float Weight { get; set; }

        public bool Recurrent { get; set; }

        public OptimizerTypes Optimizer { get; set; }

        public float? LearningRate { get; set; }

        /// <summary>
        /// Index in <see cref="NetworkDocument.SharedWeights"/> for a synapse that shares its weight; its own
        /// <see cref="Weight"/>, <see cref="Optimizer"/> and <see cref="LearningRate"/> are then ignored
        /// </summary>
        public int? SharedWeight { get; set; }
    }

    /// <summary>
    /// Weight used by several synapses (convolutional layers)
    /// </summary>
    internal sealed class SharedWeightDocument
    {
        public float Weight { get; set; }

        public OptimizerTypes Optimizer { get; set; }

        public float? LearningRate { get; set; }
    }
}
