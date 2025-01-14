﻿using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Abstractions.Interfaces.Synapses;

namespace ANN.Net.Connections
{
    internal class InactiveSynapse : BaseSynapse, ISynapse
    {
        public InactiveSynapse(INeuron input, INeuron output, IUniqueIdentityGenerator identityGenerator)
             : base(input, output, identityGenerator)
        { }

        public void Backpropagate(BackpropagateEventArgs errorSignal)
        {
            this.Input.Backpropagate(errorSignal);
        }

        public void Propagate(NeuronPropagateEventArgs value)
        {
            this.Output.Propagate(value);
        }
    }
}
