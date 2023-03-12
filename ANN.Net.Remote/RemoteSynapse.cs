using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Connections;
using ANN.Net.Remote.Abstactions.Interfaces;

namespace ANN.Net.Remote
{
    public abstract class RemoteSynapse : BaseSynapse, IRemoteSynapse
    {
        protected RemoteSynapse(INeuron input, INeuron output, IUniqueIdentityGenerator identityGenerator)
            : base(input, output, identityGenerator)
        {
        }

        public ISynapseConnection Connection { get; protected set; }

        public abstract void Backpropagate(BackpropagateEventArgs errorSignal);

        public abstract void Propagate(NeuronPropagateEventArgs value);
    }
}
