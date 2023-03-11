using System;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;

namespace ANN.Net.Connections
{
    public abstract class BaseSynapse
    {
        public BaseSynapse(INeuron input, INeuron output)
        {
            this.Input = input;
            this.Output = output;
        }


        public Action<BackpropagateEventArgs> BackpropagateConnection { get; protected set; }
        public Action<NeuronPropagateEventArgs> PropagateConnection { get; protected set; }

        protected IOptimizer optimizer;

        public Quad Weight { get; protected set; }
        public INeuron Input { get; protected set; }
        public INeuron Output { get; protected set; }


        public virtual void UpdateWeight(Quad gradient)
        {
            Weight = optimizer.Calculate(gradient, Weight);
        }
    }
}
