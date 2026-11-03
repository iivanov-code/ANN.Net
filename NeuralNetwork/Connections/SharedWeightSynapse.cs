using System;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Abstractions.Interfaces.Synapses;

namespace ANN.Net.Connections
{
    /// <summary>
    /// Synapse whose weight is a <see cref="SharedWeight"/> used by other synapses too
    /// </summary>
    [Serializable]
    internal sealed class SharedWeightSynapse : BaseSynapse, ISynapse
    {
        public SharedWeightSynapse(INeuron input, INeuron output, IUniqueIdentityGenerator identityGenerator, SharedWeight sharedWeight)
             : base(input, output, identityGenerator)
        {
            SharedWeight = sharedWeight ?? throw new ArgumentNullException(nameof(sharedWeight));
            optimizer = sharedWeight.Optimizer;
            Weight = sharedWeight.Value;
            sharedWeight.Register(this);
        }

        public SharedWeight SharedWeight { get; }

        public void Propagate(NeuronPropagateEventArgs value)
        {
            PropagateConnection(new NeuronPropagateEventArgs(value.Value * SharedWeight.Value)
            {
                ShouldActivate = value.ShouldActivate,
                ClearValue = value.ClearValue
            });
        }

        /// <summary>
        /// Passes the error on with the current shared value and reports this synapse's gradient; the shared weight
        /// changes only after every synapse using it has reported
        /// </summary>
        public void Backpropagate(BackpropagateEventArgs errorSignal)
        {
            Quad delta = errorSignal.ErrorSignal;
            Quad weighted = delta * SharedWeight.Value;

            if (Input != null)
            {
                SharedWeight.AddGradient(delta * Input.Value);
            }

            BackpropagateConnection?.Invoke(new BackpropagateEventArgs(delta, weighted, null)
            {
                ShouldActivate = errorSignal.ShouldActivate
            });
        }

        public override void UpdateWeight(Quad gradient)
        {
            SharedWeight.AddGradient(gradient);
        }

        /// <summary>
        /// Copies the shared value into <see cref="BaseSynapse.Weight"/> after an update
        /// </summary>
        internal void Synchronize()
        {
            Weight = SharedWeight.Value;
        }
    }
}
