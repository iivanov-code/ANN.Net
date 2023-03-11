using ANN.Net.Abstractions.Arguments;

namespace ANN.Net.Remote.Abstactions.Models
{
    public class RemoteBackpropagationDTO
    {
        public double ErrorSignal { get; set; }
        public float ErrorWeightedSignal { get; set; }
        public bool ShouldActivate { get; set; }


        public static implicit operator RemoteBackpropagationDTO(BackpropagateEventArgs eventArgs)
        {
            return new RemoteBackpropagationDTO
            {
                ErrorSignal = eventArgs.ErrorSignal,
                ErrorWeightedSignal = eventArgs.ErrorWeightedSignal,
                ShouldActivate = eventArgs.ShouldActivate
            };
        }
    }
}
