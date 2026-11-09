using System;
using System.Collections.Generic;

namespace ANN.Net.Abstractions.Spiking
{
    /// <summary>
    /// Spiking network with time-to-first-spike coding: every neuron fires at most once, and the earlier it fires
    /// within its layer's window of <see cref="Window"/> steps, the larger the value it stands for. Layer after layer
    /// fires in consecutive windows. A neuron whose value is (about) 0 never fires and costs nothing, so the work is at
    /// most one multiply-add per synapse, whatever the window: a longer window only gives finer values and a longer
    /// delay. A network keeps what it computed last, so it must not be used by several threads at once
    /// </summary>
    public interface IFirstSpikeNetwork
    {
        int InputCount { get; }

        int OutputCount { get; }

        /// <summary>
        /// Neurons of all layers, inputs included
        /// </summary>
        int NeuronCount { get; }

        int SynapseCount { get; }

        /// <summary>
        /// Neurons of every layer; layer 0 holds the inputs
        /// </summary>
        IReadOnlyList<int> LayerSizes { get; }

        /// <summary>
        /// Time steps of every layer's window: a neuron fires in step round(W * (1 - |value| / scale)) of it, so values
        /// are rounded to 1 / W of the layer's scale. Values below half of that do not fire
        /// </summary>
        int Window { get; set; }

        /// <summary>
        /// Presents one input with <see cref="Window"/> and decodes the outputs
        /// </summary>
        float[] Propagate(ReadOnlySpan<float> inputs);

        /// <summary>
        /// Presents one input with the given window. <see cref="SpikingRun.TimeSteps"/> is the delay until the outputs
        /// are known: one window per layer
        /// </summary>
        SpikingRun Run(ReadOnlySpan<float> inputs, int window);

        /// <summary>
        /// The spikes of the last input, in the order they were sent
        /// </summary>
        IReadOnlyList<FirstSpike> GetLastSpikes();

        /// <summary>
        /// Every synapse; a spike in the first step of its source's window adds the weight to the target in full, a
        /// later spike a share of it
        /// </summary>
        IReadOnlyList<SpikingSynapse> GetSynapses();
    }

    /// <param name="Step">Step since the input was presented</param>
    /// <param name="Sign">1, or -1 for a negative value</param>
    public readonly record struct FirstSpike(int Neuron, int Step, int Sign);
}
