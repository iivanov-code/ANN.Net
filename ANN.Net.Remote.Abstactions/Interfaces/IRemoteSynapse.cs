using ANN.Net.Abstractions.Interfaces.Synapses;

namespace ANN.Net.Remote.Abstactions.Interfaces
{
    /// <summary>
    /// Synapse whose other end lives in another process. The local neuron calls <see cref="ISynapse.Propagate"/>
    /// and the synapse forwards the value over its <see cref="Connection"/>; when the gradient of the remote part
    /// comes back, the connection calls <see cref="ISynapse.Backpropagate"/> just like a local output neuron would
    /// </summary>
    public interface IRemoteSynapse : ISynapse
    {
        /// <summary>
        /// Identity the two sides use to match values and gradients
        /// </summary>
        string Id { get; }

        ISynapseConnection Connection { get; }
    }
}
