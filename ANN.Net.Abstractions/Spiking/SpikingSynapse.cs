namespace ANN.Net.Abstractions.Spiking
{
    /// <summary>
    /// Connection between two spiking neurons. When the source fires, the weight is added to the target's input
    /// current (subtracted for a negative spike); a silent source costs nothing. A synapse into the same or an earlier
    /// layer (a recurrent or lateral one) delivers in the next time step, every other one in the same step.
    /// Neurons are numbered across the whole network: the inputs first, then layer after layer
    /// </summary>
    public readonly record struct SpikingSynapse(int Source, int Target, float Weight);
}
