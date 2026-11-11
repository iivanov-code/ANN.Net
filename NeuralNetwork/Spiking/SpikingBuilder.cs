using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Spiking;

namespace ANN.Net.Spiking
{
    /// <summary>
    /// Builds layered spiking networks from <see cref="SpikingNetworkSettings"/>
    /// </summary>
    internal static class SpikingBuilder
    {
        public static SpikingNetwork Build(SpikingNetworkSettings settings)
        {
            ArgumentNullException.ThrowIfNull(settings);
            ArgumentOutOfRangeException.ThrowIfLessThan(settings.InputCount, 1);
            ArgumentOutOfRangeException.ThrowIfLessThan(settings.TimeSteps, 1);
            if (settings.Layers.Count == 0) throw new ArgumentException("A spiking network needs at least one layer", nameof(settings));
            if (settings.InputScaling != null && settings.InputScaling.Count != settings.InputCount)
                throw new ArgumentException("The input scaling needs one value per input", nameof(settings));

            foreach (SpikingLayerSettings layer in settings.Layers)
            {
                if (layer == null) throw new ArgumentException("A layer is null", nameof(settings));
                if (layer.NeuronsCount < 1) throw new ArgumentException("Every layer needs at least one neuron", nameof(settings));
                if (!float.IsFinite(layer.MinimumWeight) || !float.IsFinite(layer.MaximumWeight) || layer.MaximumWeight < layer.MinimumWeight)
                    throw new ArgumentException("The initial weights need finite bounds with the maximum not below the minimum", nameof(settings));
                if (!(layer.ConnectionProbability >= 0 && layer.ConnectionProbability <= 1))
                    throw new ArgumentException("The connection probability must be between 0 and 1", nameof(settings));
                if (!(layer.LateralInhibition >= 0) || !float.IsFinite(layer.LateralInhibition))
                    throw new ArgumentException("The lateral inhibition must be 0 or more", nameof(settings));
            }

            settings.Plasticity?.Validate();

#pragma warning disable S2245, CA5394 // initial weights, not security
            var random = new Random(settings.Seed);

            var sizes = new List<int> { settings.InputCount };
            var activations = new List<ISpikingActivation> { settings.InputActivation ?? new IntegrateAndFireActivation() };
            var learns = new List<bool> { false };
            var synapses = new List<SpikingSynapse>();

            int previous = 0, start = settings.InputCount;
            foreach (SpikingLayerSettings layer in settings.Layers)
            {
                for (int source = previous; source < start; source++)
                {
                    for (int target = start; target < start + layer.NeuronsCount; target++)
                    {
                        if (layer.ConnectionProbability < 1 && random.NextDouble() >= layer.ConnectionProbability) continue;

                        float weight = layer.MinimumWeight + ((layer.MaximumWeight - layer.MinimumWeight) * random.NextSingle());
                        synapses.Add(new SpikingSynapse(source, target, weight));
                    }
                }

                if (layer.LateralInhibition > 0)
                {
                    for (int source = start; source < start + layer.NeuronsCount; source++)
                    {
                        for (int target = start; target < start + layer.NeuronsCount; target++)
                        {
                            if (target != source) synapses.Add(new SpikingSynapse(source, target, -layer.LateralInhibition));
                        }
                    }
                }

                sizes.Add(layer.NeuronsCount);
                activations.Add(layer.Activation ?? new IntegrateAndFireActivation(leak: 0.95f));
                learns.Add(layer.Learn);

                previous = start;
                start += layer.NeuronsCount;
            }
#pragma warning restore S2245, CA5394

            int outputs = settings.Layers[^1].NeuronsCount;
            return new SpikingNetwork(sizes, activations, learns, null, synapses, settings.InputScaling, null,
                                      SpikingReadout.Rates(start - outputs, outputs), settings.TimeSteps, settings.Plasticity);
        }

        public static SpikingNetwork Build(SpikingGraphSettings settings)
        {
            ArgumentNullException.ThrowIfNull(settings);
            ArgumentOutOfRangeException.ThrowIfLessThan(settings.TimeSteps, 1);
            if (settings.Layers == null || settings.Layers.Count < 2)
                throw new ArgumentException("A spiking network needs an input layer and at least one more", nameof(settings));
            if (settings.Layers.Any(x => x == null)) throw new ArgumentException("A layer is null", nameof(settings));
            ArgumentNullException.ThrowIfNull(settings.Synapses, nameof(settings));
            if (settings.InputScaling != null && settings.InputScaling.Count != settings.Layers[0].NeuronsCount)
                throw new ArgumentException("The input scaling needs one value per input", nameof(settings));

            int count = settings.Layers.Sum(x => x.NeuronsCount);
            float[] bias = null;
            if (settings.Bias != null)
            {
                if (settings.Bias.Count != count) throw new ArgumentException($"The bias needs one value per neuron ({count})", nameof(settings));
                if (settings.Bias.Any(x => !float.IsFinite(x))) throw new ArgumentException("The bias must be finite", nameof(settings));
                if (settings.Bias.Take(settings.Layers[0].NeuronsCount).Any(x => x != 0))
                    throw new ArgumentException("Inputs receive no bias", nameof(settings));
                bias = settings.Bias.ToArray();
            }

            int lastLayer = count - settings.Layers[^1].NeuronsCount;
            int[] outputs = settings.Outputs == null || settings.Outputs.Count == 0
                ? Enumerable.Range(lastLayer, settings.Layers[^1].NeuronsCount).ToArray()
                : settings.Outputs.ToArray();
            if (outputs.Any(x => x < 0 || x >= count)) throw new ArgumentException("An output neuron does not exist", nameof(settings));

            settings.Plasticity?.Validate();

            var activations = settings.Layers.Select(x => x.Activation ?? new IntegrateAndFireActivation()).ToList();
            var learns = settings.Layers.Select((x, i) => i > 0 && x.Learn).ToList();

            return new SpikingNetwork(settings.Layers.Select(x => x.NeuronsCount).ToList(), activations, learns, bias, settings.Synapses.ToList(),
                                      settings.InputScaling, null, SpikingReadout.Rates(outputs), settings.TimeSteps, settings.Plasticity);
        }
    }
}
