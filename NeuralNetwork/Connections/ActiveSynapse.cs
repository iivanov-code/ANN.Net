using System;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Abstractions.Interfaces.Synapses;

namespace ANN.Net.Connections
{
    [Serializable]
    internal class ActiveSynapse : BaseSynapse, ISynapse
    {
        public ActiveSynapse(INeuron input, INeuron output, IUniqueIdentityGenerator identityGenerator, IOptimizer optimizer, Quad weight)
             : base(input, output, identityGenerator)
        {
            Weight = weight;
            this.optimizer = optimizer;
        }

        /// <summary>
        /// Updates the weight with gradient * activation of the input neuron and passes the output neuron's
        /// gradient, weighted by the weight before the update, to the input neuron. The weight changes first so
        /// that nothing is left to do once the gradient has travelled on; a remote part may report completion
        /// as soon as the last gradient arrives
        /// </summary>
        public virtual void Backpropagate(BackpropagateEventArgs errorSignal)
        {
            Quad delta = errorSignal.ErrorSignal;
            Quad weighted = delta * Weight;

            if (this.Input != null)
            {
                this.UpdateWeight(delta * this.Input.Value);
            }

            if (BackpropagateConnection != null)
            {
                BackpropagateConnection(new BackpropagateEventArgs(delta, weighted, null)
                {
                    ShouldActivate = errorSignal.ShouldActivate
                });
            }
        }

        /// <summary>
        /// Sends the weighted value on as a new message, so the caller's message stays untouched for its other synapses
        /// </summary>
        public virtual void Propagate(NeuronPropagateEventArgs value)
        {
            PropagateConnection(new NeuronPropagateEventArgs(value.Value * Weight)
            {
                ShouldActivate = value.ShouldActivate,
                ClearValue = value.ClearValue
            });
        }
    }
}
