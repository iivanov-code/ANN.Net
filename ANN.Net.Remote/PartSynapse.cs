using ANN.Net.Connections;

namespace ANN.Net.Remote
{
    /// <summary>
    /// A weighted synapse of a <see cref="PartNetwork"/> with the global position of both ends.
    /// <see cref="From"/> equal to the width of the source layer means the bias neuron
    /// </summary>
    public readonly struct PartSynapse
    {
        public PartSynapse(int layer, int from, int to, BaseSynapse synapse)
        {
            this.Layer = layer;
            this.From = from;
            this.To = to;
            this.Synapse = synapse;
        }

        /// <summary>
        /// Section layer the target neuron belongs to; layer 0 is fed by the section inputs
        /// </summary>
        public int Layer { get; }

        public int From { get; }

        public int To { get; }

        public BaseSynapse Synapse { get; }
    }
}
