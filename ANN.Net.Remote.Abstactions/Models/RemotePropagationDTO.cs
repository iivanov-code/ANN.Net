using System.Linq;
using ANN.Net.Abstractions.Arguments;

namespace ANN.Net.Remote.Abstactions.Models
{
    public class RemotePropagationDTO
    {
        public bool ShouldActivate { get; set; }
        public bool ClearValue { get; set; }
        public double[] Values { get; set; }


        public static implicit operator RemotePropagationDTO(NeuronPropagateEventArgs eventArgs)
        {
            return new RemotePropagationDTO
            {
                ShouldActivate = eventArgs.ShouldActivate,
                ClearValue = eventArgs.ClearValue,
                Values = eventArgs.Values.Select(x => x.ToDouble(null)).ToArray()
            };
        }
    }
}
