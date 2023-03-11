namespace ANN.Net.Abstractions.Arguments
{
    public class RecurrentPropagateEventArgs : PropagateEventArgs
    {
        public RecurrentPropagateEventArgs(int count, bool recurrentConnection)
             : base(count)
        {
            this.RecurrentConnection = recurrentConnection;
        }

        public RecurrentPropagateEventArgs(Quad[] values, bool recurrentConnection)
             : base(values)
        {
            this.RecurrentConnection = recurrentConnection;
        }

        public bool RecurrentConnection { get; set; }
    }
}
