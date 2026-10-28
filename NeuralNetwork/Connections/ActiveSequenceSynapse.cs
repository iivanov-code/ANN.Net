using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Abstractions.Interfaces.Synapses;

namespace ANN.Net.Connections
{
    internal class ActiveSequenceSynapse : BaseSynapse, ISequenceSynapse
    {
        private Quad[] values;
        private ushort counter = 0;

        public Quad Weght { get; set; }

        public ILSTMCell OutputCell
        {
            get
            {
                return Output as ILSTMCell;
            }
        }

        public ActiveSequenceSynapse(INeuron input, INeuron output, IUniqueIdentityGenerator identityGenerator, ushort count)
             : base(input, output, identityGenerator)
        {
            values = new Quad[count];
        }

        public void Backpropagate(BackpropagateEventArgs errorSignal)
        {
        }

        public void Propagate(SequencePropagateEventArgs values)
        {
            OutputCell.Propagate(new NeuronPropagateEventArgs
            {
                Values = values.Values,
                ClearValue = values.ClearValue
            });
        }

        public void Propagate(NeuronPropagateEventArgs value)
        {
            counter++;
            if (counter < values.Length)
            {
                values[counter - 1] = value.Value * Weght;
            }
            else if (counter == values.Length)
            {
                counter = 0;
                Propagate(new SequencePropagateEventArgs(this.values)
                {
                    ClearValue = value.ClearValue
                });
            }
        }
    }
}
