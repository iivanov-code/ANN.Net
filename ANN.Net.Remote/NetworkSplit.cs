using ANN.Net.Abstractions.Settings;

namespace ANN.Net.Remote
{
    /// <summary>
    /// One network description cut in two: the head keeps the input layer, the hidden layers up to and
    /// including the boundary layer (built as its output layer), the tail gets an identity input layer of the
    /// boundary's size, the remaining hidden layers and the real output layer
    /// </summary>
    public sealed class NetworkSplit
    {
        public NetworkSettings HeadSettings { get; set; }

        public RecurrentLayerSettings[] HeadHiddenLayers { get; set; }

        public NetworkSettings TailSettings { get; set; }

        public RecurrentLayerSettings[] TailHiddenLayers { get; set; }

        /// <summary>
        /// Index of the hidden layer whose outgoing synapses cross the wire
        /// </summary>
        public int BoundaryLayer { get; set; }

        public int BoundaryNeurons { get; set; }
    }
}
