using System.ComponentModel;
using ANN.Net.Abstractions.Enums;

namespace ANN.Net.Abstractions.Settings
{
    /// <summary>
    /// A stack of recurrent layers (LSTM, LSTM with peepholes, GRU or plain tanh RNN) and a linear output read from the
    /// last layer at every step
    /// </summary>
    public class RecurrentNetworkSettings
    {
        /// <summary>
        /// Size of every input vector
        /// </summary>
        public ushort InputDimension { get; set; }

        /// <summary>
        /// Size of the hidden state (and of the cell state of an LSTM) of every layer
        /// </summary>
        public ushort HiddenDimension { get; set; }

        /// <summary>
        /// Number of stacked recurrent layers; every layer reads the hidden states of the one below
        /// </summary>
        [DefaultValue(1)]
        public ushort LayersCount { get; set; } = 1;

        /// <summary>
        /// Size of every output vector
        /// </summary>
        public ushort OutputDimension { get; set; }

        /// <summary>
        /// <see cref="NeuronType.LSTM"/>, <see cref="NeuronType.LSTMPeepHole"/>, <see cref="NeuronType.GRU"/> or
        /// <see cref="NeuronType.RNN"/> (h = tanh(W [x, h] + b))
        /// </summary>
        [DefaultValue(NeuronType.LSTM)]
        public NeuronType CellType { get; set; } = NeuronType.LSTM;

        /// <summary>
        /// Soft-max on every output vector, trained with cross entropy; otherwise the outputs are linear and trained with
        /// the squared error
        /// </summary>
        public bool OutputApplySoftMax { get; set; }

        /// <summary>
        /// Initial bias of the gates that keep the state: the forget gates of an LSTM and the update gates of a GRU.
        /// 1 keeps most of the state from the start, so gradients reach far back in time before the network has learned
        /// to keep anything
        /// </summary>
        [DefaultValue(1f)]
        public float ForgetGateBias { get; set; } = 1;

        /// <summary>
        /// Largest L2 norm of the gradient of all weights for one sequence; a larger gradient is scaled down to it. 0 for
        /// no clipping. Recurrent networks can have exploding gradients, and clipping keeps one bad sequence from
        /// throwing the weights away
        /// </summary>
        public float GradientClipping { get; set; }

        [DefaultValue(OptimizerTypes.Normal)]
        public OptimizerTypes OptimizerType { get; set; }

        [DefaultValue(WeightInitType.Xavier)]
        public WeightInitType InitializationType { get; set; }

        /// <summary>
        /// Seed of the initial weights; the same seed builds the same network. Null draws them from a shared random source
        /// </summary>
        public int? Seed { get; set; }
    }
}
