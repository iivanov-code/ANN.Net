using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Remote.Abstactions.Interfaces;

namespace ANN.Net.Remote
{
    public class ReceiverRemoteSynapse : RemoteSynapse
    {
        public ReceiverRemoteSynapse(INeuron input, ISynapseConnection connection)
            : base(input, null)
        {
            this.Connection = connection;
        }

        public override void Backpropagate(BackpropagateEventArgs errorSignal)
        {
            this.Connection.Backpropagate(errorSignal);
        }

        public override void Propagate(NeuronPropagateEventArgs value)
        {
            PropagateConnection(value.ApplyWeight(Weight));
        }
    }
}
