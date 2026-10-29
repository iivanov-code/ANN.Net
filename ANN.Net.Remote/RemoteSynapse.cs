using System;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Connections;
using ANN.Net.Remote.Abstactions.Interfaces;
using ANN.Net.Remote.Abstactions.Models;

namespace ANN.Net.Remote
{
    /// <summary>
    /// Sending end of a connection that crosses to another process: the axon of a boundary neuron.
    /// It carries no weight; the weights sit in the first layer of the remote part, exactly where the
    /// synapses of the next layer would sit in a local network. Forward it hands the neuron's value to the
    /// connection, backward the connection calls <see cref="Backpropagate"/> with the remote gradient and
    /// the synapse passes it on to its neuron unchanged
    /// </summary>
    public sealed class RemoteSynapse : BaseSynapse, IRemoteSynapse
    {
        public RemoteSynapse(INeuron input, ISynapseConnection connection, IUniqueIdentityGenerator identityGenerator)
            : base(input, null, identityGenerator)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (connection == null) throw new ArgumentNullException(nameof(connection));

            this.Connection = connection;
            this.Weight = 1;
            connection.Attach(this);
        }

        public ISynapseConnection Connection { get; }

        /// <summary>
        /// Called by the local neuron with its activated value
        /// </summary>
        public void Propagate(NeuronPropagateEventArgs value)
        {
            if (value == null) throw new ArgumentNullException(nameof(value));

            this.Connection.Propagate(new RemotePropagationDTO
            {
                Id = this.Id,
                ShouldActivate = value.ShouldActivate,
                ClearValue = value.ClearValue,
                Values = new[] { value.Value.ToDouble(null) }
            });
        }

        /// <summary>
        /// Called by the connection with the gradient of the remote part with respect to the neuron's value
        /// </summary>
        public void Backpropagate(BackpropagateEventArgs errorSignal)
        {
            if (errorSignal == null) throw new ArgumentNullException(nameof(errorSignal));

            Quad gradient = errorSignal.ErrorSignal;
            this.BackpropagateConnection(new BackpropagateEventArgs(gradient, gradient, null)
            {
                ShouldActivate = errorSignal.ShouldActivate
            });
        }

        /// <summary>
        /// There is no weight on this side of the wire
        /// </summary>
        public override void UpdateWeight(Quad gradient)
        { }
    }
}
