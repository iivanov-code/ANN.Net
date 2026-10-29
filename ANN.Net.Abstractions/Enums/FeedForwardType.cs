namespace ANN.Net.Abstractions.Enums
{
    /// <summary>
    /// Implementation of the position-wise feed forward sub-layer of a transformer block
    /// </summary>
    public enum FeedForwardType
    {
        /// <summary>
        /// Two dense projections over weight matrices; weights are updated once per sequence
        /// </summary>
        Matrix = 0,

        /// <summary>
        /// A graph of neuron objects connected by synapses, built with <c>NetworkBuilder</c>;
        /// weights are updated after every position of the sequence
        /// </summary>
        NeuronGraph = 1
    }
}
