using System;
using System.Collections.Generic;

namespace ANN.Net.Abstractions.Spiking
{
    /// <summary>
    /// Network of spiking neurons simulated in discrete time steps. Each input is normalized and presented as a
    /// constant current to the input neurons for <see cref="TimeSteps"/> steps; in every step every layer adds up the
    /// weights of the synapses whose source fired, and its <see cref="ISpikingActivation"/> decides which neurons fire.
    /// Work is done only for spikes, so silent neurons cost nothing beyond their state update.
    /// A network keeps the state of its neurons, so it must not be used by several threads at once
    /// </summary>
    public interface ISpikingNetwork
    {
        int InputCount { get; }

        int OutputCount { get; }

        /// <summary>
        /// Neurons of all layers, inputs included
        /// </summary>
        int NeuronCount { get; }

        int SynapseCount { get; }

        /// <summary>
        /// Neurons of every layer; layer 0 holds the inputs, the last one the outputs
        /// </summary>
        IReadOnlyList<int> LayerSizes { get; }

        /// <summary>
        /// Time steps <see cref="Propagate"/> and <see cref="Learn"/> present every input for
        /// </summary>
        int TimeSteps { get; set; }

        /// <summary>
        /// Learning rule of <see cref="Learn"/>
        /// </summary>
        StdpOptions Plasticity { get; set; }

        /// <summary>
        /// Spike function of a layer; layer 0 turns the normalized inputs into spikes
        /// </summary>
        ISpikingActivation GetActivation(int layer);

        /// <summary>
        /// Replaces the spike function of every neuron of a layer. The neurons keep their state, which the new
        /// function interprets its own way, so it is best done between inputs
        /// </summary>
        void SetActivation(int layer, ISpikingActivation activation);

        /// <summary>
        /// Presents one input for <see cref="TimeSteps"/> steps and decodes the outputs
        /// </summary>
        float[] Propagate(ReadOnlySpan<float> inputs);

        /// <summary>
        /// Presents one input, starting from rest, for the given number of steps
        /// </summary>
        /// <param name="learn">Change the weights with <see cref="Plasticity"/> while the input is presented</param>
        SpikingRun Run(ReadOnlySpan<float> inputs, int timeSteps, bool learn = false);

        /// <summary>
        /// Presents one input for <see cref="TimeSteps"/> steps and changes the weights with <see cref="Plasticity"/>
        /// (unsupervised learning)
        /// </summary>
        SpikingRun Learn(ReadOnlySpan<float> inputs);

        /// <summary>
        /// Every synapse with its current weight
        /// </summary>
        IReadOnlyList<SpikingSynapse> GetSynapses();
    }
}
