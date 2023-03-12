using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Remote.Abstactions.Interfaces;
using ANN.Net.Remote.Abstactions.Models;

namespace ANN.Net.Remote
{
    public class ReceiverRemoteSynapse : RemoteSynapse
    {
        public ReceiverRemoteSynapse(INeuron input, ISynapseConnection connection, IUniqueIdentityGenerator identityGenerator)
            : base(input, null, identityGenerator)
        {
            this.Connection = connection;
        }

        public override void Backpropagate(BackpropagateEventArgs errorSignal)
        {
            RemoteBackpropagationDTO remoteErrorSignal = (RemoteBackpropagationDTO)errorSignal;
            remoteErrorSignal.Id = this.Id;
            this.Connection.Backpropagate(remoteErrorSignal);
        }

        public override void Propagate(NeuronPropagateEventArgs value)
        {
            PropagateConnection(value.ApplyWeight(Weight));
        }
    }
}
