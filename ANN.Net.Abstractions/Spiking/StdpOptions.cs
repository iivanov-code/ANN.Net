using System;

namespace ANN.Net.Abstractions.Spiking
{
    /// <summary>
    /// Spike-timing-dependent plasticity: a synapse whose source fires shortly before its target is strengthened,
    /// one whose source fires shortly after its target is weakened. Only the feed-forward synapses into layers with
    /// <see cref="SpikingLayerSettings.Learn"/> change. Every neuron keeps two traces that jump by 1 when it fires and
    /// decay exponentially, so an update needs only the two neurons of the synapse
    /// </summary>
    public sealed class StdpOptions
    {
        /// <summary>
        /// Weight change when the target fires, times the source's trace
        /// </summary>
        public float PotentiationRate { get; set; } = 0.01f;

        /// <summary>
        /// Weight change when the source fires, times the target's trace. Slightly larger than the potentiation so
        /// that inputs firing at random end up weaker than inputs that cause spikes
        /// </summary>
        public float DepressionRate { get; set; } = 0.0105f;

        /// <summary>
        /// Time steps in which the source trace falls to 1/e
        /// </summary>
        public float PotentiationTimeConstant { get; set; } = 20;

        /// <summary>
        /// Time steps in which the target trace falls to 1/e
        /// </summary>
        public float DepressionTimeConstant { get; set; } = 20;

        public float MinimumWeight { get; set; }

        public float MaximumWeight { get; set; } = 1;

        /// <summary>
        /// Scales every change by the distance to the bound it moves towards, so weights approach the bounds slowly
        /// and spread out instead of piling up at them. Otherwise weights are clamped to the bounds
        /// </summary>
        public bool SoftBounds { get; set; } = true;

        /// <summary>
        /// After every learned input the learning synapses into each neuron are scaled so that their weights add up to
        /// this value: no neuron can win every input by growing all its weights. 0 leaves them as they are
        /// </summary>
        public float IncomingWeightSum { get; set; }

        public void Validate()
        {
            if (!(PotentiationRate >= 0) || !(DepressionRate >= 0))
                throw new ArgumentOutOfRangeException(nameof(PotentiationRate), "The learning rates must be 0 or more");
            if (!(PotentiationTimeConstant > 0) || !(DepressionTimeConstant > 0))
                throw new ArgumentOutOfRangeException(nameof(PotentiationTimeConstant), "The time constants must be greater than 0");
            if (!(MaximumWeight > MinimumWeight))
                throw new ArgumentOutOfRangeException(nameof(MaximumWeight), "The maximum weight must be greater than the minimum weight");
            if (!(IncomingWeightSum >= 0))
                throw new ArgumentOutOfRangeException(nameof(IncomingWeightSum), "The sum must be 0 or more");
        }
    }
}
