using System;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Abstractions.Interfaces.Synapses;

namespace ANN.Net.Connections
{
    [Serializable]
    internal class ActiveSynapse : BaseSynapse, ISynapse
    {
        public ActiveSynapse(INeuron input, INeuron output, IOptimizer optimizer, Quad weight)
             : base(input, output)
        {
            Weight = weight;
            PropagateConnection = output.Propagate;
            BackpropagateConnection = input.Backpropagate;
            this.optimizer = optimizer;
        }

        public virtual void Backpropagate(BackpropagateEventArgs errorSignal)
        {
            errorSignal.ErrorWeightedSignal = errorSignal.ErrorSignal * Weight;
            errorSignal.UpdateWeightAction = this.UpdateWeight;
            BackpropagateConnection(errorSignal);
        }

        public virtual void Propagate(NeuronPropagateEventArgs value)
        {
            PropagateConnection(value.ApplyWeight(Weight));
        }
    }
}
