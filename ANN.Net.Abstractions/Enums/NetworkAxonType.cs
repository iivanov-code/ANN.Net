namespace ANN.Net.Abstractions.Enums
{
    public enum NetworkAxonType
    {
        /// <summary>
        /// Second half of the synapse or the one that
        /// receives the information while propagating forward
        /// </summary>
        Receiver,

        /// <summary>
        /// First half of the synapse or the one that
        /// sends the information while propagating forward
        /// </summary>
        Sender
    }
}
