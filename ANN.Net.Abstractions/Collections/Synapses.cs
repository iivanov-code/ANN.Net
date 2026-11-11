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

        /// <summary>
        /// Adds one incoming weighted value. Once every synapse has delivered, the sum is handed
        /// to the neuron and the collection is reset for the next propagation
        /// </summary>
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
                    Quad sum = Value;
                    Value = 0;

                    sendPropagateValue(new NeuronPropagateEventArgs(sum)
                    {
                        ShouldActivate = value.ShouldActivate,
                        ClearValue = value.ClearValue
                    });
                }
            }
        }

        /// <summary>
        /// Adds one incoming weighted error. Once every forward synapse has delivered, the sum is handed
        /// to the neuron and the collection is reset for the next backpropagation
        /// </summary>
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
                    Quad sum = Value;
                    Value = 0;

                    sendBackpropagateValue(new BackpropagateEventArgs(sum)
                    {
                        ShouldActivate = errorSignal.ShouldActivate
                    });
                }
            }
        }
    }
}
