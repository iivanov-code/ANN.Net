using System.ComponentModel;
using ANN.Net.Abstractions.Enums;

namespace ANN.Net.Abstractions.Settings
{
    public class TransformerSettings
    {
        /// <summary>
        /// Size of every vector in the input sequence (features or token embedding)
        /// </summary>
        public ushort InputDimension { get; set; }

        /// <summary>
        /// Size of the vectors flowing through the encoder blocks (d_model)
        /// </summary>
        public ushort ModelDimension { get; set; }

        /// <summary>
        /// Number of attention heads. ModelDimension must be divisible by it
        /// </summary>
        public ushort HeadsCount { get; set; }

        /// <summary>
        /// Size of the hidden layer inside the position-wise feed forward network
        /// </summary>
        public ushort FeedForwardDimension { get; set; }

        /// <summary>
        /// Number of stacked encoder blocks
        /// </summary>
        public ushort LayersCount { get; set; }

        /// <summary>
        /// Size of every vector in the output sequence
        /// </summary>
        public ushort OutputDimension { get; set; }

        /// <summary>
        /// Longest sequence the network accepts; defines the positional encoding table
        /// </summary>
        public ushort MaxSequenceLength { get; set; }

        /// <summary>
        /// When true every position attends only to itself and the positions before it
        /// </summary>
        public bool UseCausalMask { get; set; }

        public bool OutputApplySoftMax { get; set; }

        [DefaultValue(ActivationTypes.RELU)]
        public ActivationTypes FeedForwardActivationType { get; set; } = ActivationTypes.RELU;

        /// <summary>
        /// Implementation of the feed forward sub-layer: dense weight matrices, or a graph of neuron objects
        /// </summary>
        [DefaultValue(FeedForwardType.Matrix)]
        public FeedForwardType FeedForwardType { get; set; }

        [DefaultValue(OptimizerTypes.Normal)]
        public OptimizerTypes OptimizerType { get; set; }

        [DefaultValue(WeightInitType.Xavier)]
        public WeightInitType InitializationType { get; set; }
    }
}
