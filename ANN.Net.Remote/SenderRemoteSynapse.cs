using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Remote.Abstactions.Interfaces;
using ANN.Net.Remote.Abstactions.Models;

namespace ANN.Net.Remote
{
    public class SenderRemoteSynapse : RemoteSynapse
    {
        public SenderRemoteSynapse(INeuron input, ISynapseConnection connection, IUniqueIdentityGenerator identityGenerator)
            : base(input, null, identityGenerator)
        {
            this.Connection = connection;
        }

        public override void Backpropagate(BackpropagateEventArgs errorSignal)
        {
            errorSignal.ErrorWeightedSignal = errorSignal.ErrorSignal * Weight;
            errorSignal.UpdateWeightAction = this.UpdateWeight;
            BackpropagateConnection(errorSignal);
        }

        public override void Propagate(NeuronPropagateEventArgs value)
        {
            RemotePropagationDTO remoteValue = (RemotePropagationDTO)value;
            remoteValue.Id = this.Id;

            this.Connection.Propagate(remoteValue);
        }
    }
}
