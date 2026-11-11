namespace ANN.Net.Abstractions.Arguments
{
    public class NeuronPropagateEventArgs : PropagateEventArgs
    {
        public NeuronPropagateEventArgs()
             : base(1)
        {
            this.ShouldActivate = true;
        }

        public NeuronPropagateEventArgs(Quad value)
             : this()
        {
            this.Value = value;
        }

        public bool ShouldActivate { get; set; }

        public Quad Value
        {
            get
            {
                return Values[0];
            }
            set
            {
                Values[0] = value;
            }
        }

        public NeuronPropagateEventArgs ApplyWeight(Quad multiplier)
        {
            Value *= multiplier;
            return this;
        }

        public NeuronPropagateEventArgs SumValue(Quad value)
        {
            Value += value;
            return this;
        }
    }
}
