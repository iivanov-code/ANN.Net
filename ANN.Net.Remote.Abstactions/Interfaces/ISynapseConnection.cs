using ANN.Net.Remote.Abstactions.Models;

namespace ANN.Net.Remote.Abstactions.Interfaces
{
    /// <summary>
    /// Link shared by every remote synapse that crosses the same boundary. One propagation is complete,
    /// and is sent to the remote part as one message, once every attached synapse has delivered its value
    /// </summary>
    public interface ISynapseConnection
    {
        void Attach(IRemoteSynapse synapse);

        /// <summary>
        /// Hands over the value of one synapse for the current propagation
        /// </summary>
        void Propagate(RemotePropagationDTO remoteData);
    }
}
