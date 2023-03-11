using System;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Synapses;

namespace ANN.Net.Abstractions.Collections
{
    public class Synapses : SynapseCollection<ISynapse>, ISynapses
    {
        public Synapses()
             : base()
        { }

        public Synapses(Action<NeuronPropagateEventArgs> receivingFunc)
             : base()
        {
            this.sendPropagateValue = receivingFunc;
        }

        public Synapses(Action<BackpropagateEventArgs> receivingFunc)
            : base()
        {
            this.sendBackpropagateValue = receivingFunc;
        }

        private object padlock = new object();
        private Action<NeuronPropagateEventArgs> sendPropagateValue;
        private Action<BackpropagateEventArgs> sendBackpropagateValue;

        public void PropagateForEach(NeuronPropagateEventArgs value)
        {
            foreach (var synapse in this.list)
            {
                synapse.Propagate(value);
            }
        }

        public void BackpropagateForEach(BackpropagateEventArgs errorSignal)
        {
            foreach (var synapse in this.list)
            {
                synapse.Backpropagate(errorSignal);
            }
        }

        public void AccountSignal(NeuronPropagateEventArgs value)
        {
            lock (padlock)
            {
                Value += value.Value;

                checked
                {
                    activatedCount++;
                }

                if (activatedCount == this.list.Count)
                {
                    activatedCount = 0;
                    value.Value = Value;
                    sendPropagateValue(value);
                }
            }
        }

        public void AccumulateError(BackpropagateEventArgs errorSignal)
        {
            lock (padlock)
            {
                Value += errorSignal.ErrorWeightedSignal;

                checked
                {
                    activatedCount++;
                }

                if (activatedCount + recurrentCount == this.list.Count)
                {
                    activatedCount = 0;
                    sendBackpropagateValue(new BackpropagateEventArgs(Value));
                }
            }

            errorSignal.UpdateWeight(errorSignal.ErrorSignal * Value); //Gradient
        }
    }
}
