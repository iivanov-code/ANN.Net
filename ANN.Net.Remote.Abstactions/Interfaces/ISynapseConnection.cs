using ANN.Net.Remote.Abstactions.Models;

namespace ANN.Net.Remote.Abstactions.Interfaces
{
    public interface ISynapseConnection
    {
        void Backpropagate(RemoteBackpropagationDTO remoteData);
        void Propagate(RemotePropagationDTO remoteData);
    }
}
