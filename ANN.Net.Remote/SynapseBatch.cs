using System;
using System.Collections.Generic;
using ANN.Net.Remote.Abstactions.Interfaces;
using ANN.Net.Remote.Abstactions.Models;

namespace ANN.Net.Remote
{
    /// <summary>
    /// Collects one value per attached synapse and hands the complete set over once; delivers gradients back by index
    /// </summary>
    internal sealed class SynapseBatch
    {
        private readonly List<IRemoteSynapse> synapses;
        private readonly Dictionary<string, int> indexById;
        private readonly object padlock;
        private RemotePropagationDTO[] pending;
        private int delivered;

        public SynapseBatch()
        {
            this.synapses = new List<IRemoteSynapse>();
            this.indexById = new Dictionary<string, int>(StringComparer.Ordinal);
            this.padlock = new object();
        }

        public IReadOnlyList<IRemoteSynapse> Synapses => this.synapses;

        public int Count => this.synapses.Count;

        public bool HasPendingSignals
        {
            get
            {
                lock (this.padlock)
                {
                    return this.delivered > 0;
                }
            }
        }

        public void Attach(IRemoteSynapse synapse)
        {
            if (synapse == null) throw new ArgumentNullException(nameof(synapse));
            if (string.IsNullOrEmpty(synapse.Id)) throw new ArgumentException("A remote synapse needs an id", nameof(synapse));

            lock (this.padlock)
            {
                if (this.delivered > 0) throw new InvalidOperationException("Cannot attach a synapse while a propagation is in progress");
                if (this.indexById.ContainsKey(synapse.Id)) throw new ArgumentException($"A synapse with id {synapse.Id} is already attached", nameof(synapse));

                this.indexById.Add(synapse.Id, this.synapses.Count);
                this.synapses.Add(synapse);
                this.pending = null;
            }
        }

        /// <summary>
        /// Stores one value; returns true, and the complete set, once every synapse has delivered
        /// </summary>
        public bool Add(RemotePropagationDTO remoteData, out RemotePropagationDTO[] batch)
        {
            if (remoteData == null) throw new ArgumentNullException(nameof(remoteData));
            if (remoteData.Values == null || remoteData.Values.Length != 1)
                throw new ArgumentException("A remote synapse carries exactly one value", nameof(remoteData));

            lock (this.padlock)
            {
                if (remoteData.Id == null || !this.indexById.TryGetValue(remoteData.Id, out int index))
                    throw new ArgumentException($"Synapse {remoteData.Id} is not attached to this connection", nameof(remoteData));

                if (this.pending == null)
                {
                    this.pending = new RemotePropagationDTO[this.synapses.Count];
                }

                if (this.pending[index] != null)
                    throw new InvalidOperationException($"Synapse {remoteData.Id} delivered twice in one propagation");

                this.pending[index] = remoteData;
                this.delivered++;

                if (this.delivered < this.synapses.Count)
                {
                    batch = null;
                    return false;
                }

                batch = this.pending;
                this.pending = null;
                this.delivered = 0;
                return true;
            }
        }

        public int IndexOf(string id)
        {
            lock (this.padlock)
            {
                return id != null && this.indexById.TryGetValue(id, out int index) ? index : -1;
            }
        }

        /// <summary>
        /// Calls <see cref="Abstractions.Interfaces.Synapses.ISynapse.Backpropagate"/> on every synapse with its gradient, in attachment order
        /// </summary>
        public void DeliverGradients(Quad[] gradients)
        {
            if (gradients == null) throw new ArgumentNullException(nameof(gradients));
            if (gradients.Length != this.synapses.Count)
                throw new ArgumentException($"Expected {this.synapses.Count} gradients but received {gradients.Length}", nameof(gradients));

            for (int i = 0; i < gradients.Length; i++)
            {
                this.Deliver(i, gradients[i]);
            }
        }

        public void Deliver(int index, Quad gradient)
        {
            this.synapses[index].Backpropagate(new Abstractions.Arguments.BackpropagateEventArgs(gradient, gradient, null));
        }

        public static Quad[] Values(RemotePropagationDTO[] batch)
        {
            Quad[] values = new Quad[batch.Length];
            for (int i = 0; i < batch.Length; i++)
            {
                values[i] = (Quad)batch[i].Values[0];
            }

            return values;
        }
    }
}
