using ANN.Net.Abstractions.Arguments;
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

        public ActiveSequenceSynapse(INeuron input, INeuron output, ushort count)
             : base(input, output)
        {
            values = new Quad[count];
        }

        public void Backpropagate(BackpropagateEventArgs errorSignal)
        {
        }

        public void Propagate(CellPropagateEventArgs values)
        {
            OutputCell.Propagate(values);
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
                var values = new CellPropagateEventArgs(this.values, null, false);
                values.ClearValue = value.ClearValue;
                OutputCell.Propagate(values);
            }
        }
    }
}
