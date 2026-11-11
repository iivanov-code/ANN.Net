using System;
using ANN.Net.Abstractions.Spiking;

namespace ANN.Net.Spiking
{
    /// <summary>
    /// Random input encoding: in every step the neuron fires with the probability |current| (at most 1), a negative
    /// spike for a negative current. It keeps no potential, so its spikes are irregular like those of real sensory
    /// neurons; the integrate-and-fire encoding fires the same rate evenly spaced. Meant for layer 0
    /// </summary>
    public sealed class PoissonActivation : ISpikingActivation
    {
        private readonly Random random;

        public PoissonActivation(int seed)
        {
            Seed = seed;
#pragma warning disable S2245, CA5394 // spike trains, not security
            random = new Random(seed);
#pragma warning restore S2245, CA5394
        }

        public int Seed { get; }

        public int Step(ref SpikingNeuronState neuron, float current)
        {
#pragma warning disable CA5394
            if (current == 0 || random.NextSingle() >= Math.Abs(current)) return 0;
#pragma warning restore CA5394
            return current > 0 ? 1 : -1;
        }

        public override string ToString() => $"Poisson (seed {Seed})";
    }
}
