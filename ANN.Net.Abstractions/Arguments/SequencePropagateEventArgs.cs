namespace ANN.Net.Abstractions.Arguments
{
    public class SequencePropagateEventArgs : PropagateEventArgs
    {
        public SequencePropagateEventArgs(int count)
             : base(count)
        { }

        public SequencePropagateEventArgs(Quad[] values)
            : base(values)
        { }
    }
}
