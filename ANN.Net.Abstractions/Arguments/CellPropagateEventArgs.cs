namespace ANN.Net.Abstractions.Arguments
{
    public class CellPropagateEventArgs : RecurrentPropagateEventArgs
    {
        public CellPropagateEventArgs(int count, bool recurrentConnecton)
             : base(count, recurrentConnecton)
        {
            this.PrevCellState = new Quad[count];
        }

        public CellPropagateEventArgs(Quad[] values, Quad[] prevCellStateValues, bool recurrentConnection)
             : base(values, recurrentConnection)
        {
            this.PrevCellState = prevCellStateValues;
        }

        public Quad[] PrevCellState { get; set; }
    }
}
