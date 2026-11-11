using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.Connections
{
    /// <summary>
    /// Weight used by several synapses, as in a convolution where one kernel weight connects many positions. Every
    /// synapse reports its gradient during backpropagation; once all of them have, the summed gradient goes through the
    /// optimizer in a single step. Until then every synapse keeps using the value before the update
    /// </summary>
    [Serializable]
    internal sealed class SharedWeight
    {
        private readonly object padlock = new object();
        private readonly List<SharedWeightSynapse> synapses = new List<SharedWeightSynapse>();
        private Quad gradient;
        private int reported;

        public SharedWeight(Quad value, IOptimizer optimizer)
        {
            Value = value;
            Optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));
        }

        public Quad Value { get; private set; }

        public IOptimizer Optimizer { get; }

        public int UsersCount => synapses.Count;

        internal void Register(SharedWeightSynapse synapse)
        {
            synapses.Add(synapse);
        }

        /// <summary>
        /// Adds the gradient of one synapse; the last one of a backpropagation applies the optimizer
        /// </summary>
        public void AddGradient(Quad synapseGradient)
        {
            lock (padlock)
            {
                gradient += synapseGradient;
                if (++reported < synapses.Count) return;

                Value = Optimizer.Calculate(gradient, Value);
                gradient = 0;
                reported = 0;

                foreach (SharedWeightSynapse synapse in synapses)
                {
                    synapse.Synchronize();
                }
            }
        }
    }
}
