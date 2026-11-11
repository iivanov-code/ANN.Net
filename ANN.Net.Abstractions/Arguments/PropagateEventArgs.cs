using System;

namespace ANN.Net.Abstractions.Arguments
{
    public class PropagateEventArgs : EventArgs
    {
        public PropagateEventArgs(int count)
        {
            Values = new Quad[count];
        }

        public PropagateEventArgs(Quad[] values)
        {
            Values = values;
        }

        public bool ClearValue { get; set; }
        public Quad[] Values { get; set; }
    }
}
