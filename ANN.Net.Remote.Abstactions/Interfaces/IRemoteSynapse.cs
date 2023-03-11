using ANN.Net.Abstractions.Interfaces.Synapses;

namespace ANN.Net.Remote.Abstactions.Interfaces
{
    public interface IRemoteSynapse : ISynapse
    {
        ISynapseConnection Connection { get; }
    }
}
