using System;
using ANN.Net.Remote.Abstactions.Interfaces;
using ANN.Net.Remote.Abstactions.Models;

namespace ANN.Net.Remote
{
    /// <summary>
    /// Boundary that hands the complete set of values to a callback instead of a remote call; the owner decides
    /// where they go. Gradients come back through <see cref="DeliverGradients"/>, in attachment order
    /// </summary>
    public sealed class CallbackSynapseConnection : ISynapseConnection
    {
        private readonly SynapseBatch batch;
        private readonly Action<Quad[]> onComplete;

        public CallbackSynapseConnection(Action<Quad[]> onComplete)
        {
            this.onComplete = onComplete ?? throw new ArgumentNullException(nameof(onComplete));
            this.batch = new SynapseBatch();
        }

        public int Count => this.batch.Count;

        public void Attach(IRemoteSynapse synapse)
        {
            this.batch.Attach(synapse);
        }

        public void Propagate(RemotePropagationDTO remoteData)
        {
            if (this.batch.Add(remoteData, out RemotePropagationDTO[] complete))
            {
                this.onComplete(SynapseBatch.Values(complete));
            }
        }

        public void DeliverGradients(Quad[] gradients)
        {
            this.batch.DeliverGradients(gradients);
        }
    }
}
