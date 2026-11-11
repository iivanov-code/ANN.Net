using System;

namespace ANN.Net.Abstractions.Arguments
{
    public class BackpropagateEventArgs : EventArgs
    {
        private Action<Quad> updateWeight;

        public BackpropagateEventArgs(Quad errorSignal)
        {
            this.ErrorSignal = errorSignal;
            this.ErrorWeightedSignal = 0;
            updateWeight = null;
            this.ShouldActivate = true;
        }

        public BackpropagateEventArgs(Quad errorSignal, float eWeightedSignal, Action<Quad> updateWeight)
        {
            this.ErrorSignal = errorSignal;
            this.ErrorWeightedSignal = eWeightedSignal;
            this.updateWeight = updateWeight;
            this.ShouldActivate = true;
        }

        public Quad ErrorSignal { get; set; }
        public float ErrorWeightedSignal { get; set; }
        public bool ShouldActivate { get; set; }

        public Action<Quad> UpdateWeightAction
        {
            set
            {
                this.updateWeight = value;
            }
        }

        public void UpdateWeight(Quad gradient)
        {
            if (updateWeight != null)
                updateWeight(gradient); //Gradient
        }
    }
}
