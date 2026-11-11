using System;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;

namespace ANN.Net.Connections
{
    public abstract class BaseSynapse
    {
        /// <param name="idGenerator">Only remote synapses take an identity from it; local synapses need none, which saves
        /// a string per synapse</param>
        public BaseSynapse(INeuron input, INeuron output, IUniqueIdentityGenerator idGenerator)
        {
            this.Input = input;
            this.Output = output;
        }

        /// <summary>
        /// Backpropagate of the input neuron. Synapses call the neuron directly; no delegate is stored per synapse
        /// </summary>
        public Action<BackpropagateEventArgs> BackpropagateConnection => Input == null ? null : Input.Backpropagate;

        /// <summary>
        /// Propagate of the output neuron. Synapses call the neuron directly; no delegate is stored per synapse
        /// </summary>
        public Action<NeuronPropagateEventArgs> PropagateConnection => Output == null ? null : Output.Propagate;

        protected IOptimizer optimizer;

        internal IOptimizer Optimizer => optimizer;

        public Quad Weight { get; protected internal set; }
        public INeuron Input { get; protected set; }
        public INeuron Output { get; protected set; }


        public virtual void UpdateWeight(Quad gradient)
        {
            Weight = optimizer.Calculate(gradient, Weight);
        }
    }
}
