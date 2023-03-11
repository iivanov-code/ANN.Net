using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Remote.Abstactions.Interfaces;

namespace ANN.Net.Remote
{
    public class SenderRemoteSynapse : RemoteSynapse
    {
        public SenderRemoteSynapse(INeuron input, ISynapseConnection connection)
            : base(input, null)
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
            this.Connection.Propagate(value);
        }
    }
}
