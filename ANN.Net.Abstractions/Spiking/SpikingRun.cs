using System.Collections.Generic;

namespace ANN.Net.Abstractions.Spiking
{
    /// <summary>
    /// Result of presenting one input to a spiking network
    /// </summary>
    /// <param name="Outputs">Decoded outputs: the firing rates of the output neurons, or for a converted network the
    /// values the original network would compute</param>
    /// <param name="OutputSpikes">Spikes of every output neuron, negative spikes subtracted</param>
    /// <param name="Spikes">Spikes of all neurons, inputs included</param>
    /// <param name="SynapticEvents">Synapse deliveries, one addition each: the work a dense network would do with one
    /// multiplication and addition per synapse</param>
    public sealed record SpikingRun(IReadOnlyList<float> Outputs, IReadOnlyList<int> OutputSpikes, int TimeSteps, long Spikes, long SynapticEvents);
}
