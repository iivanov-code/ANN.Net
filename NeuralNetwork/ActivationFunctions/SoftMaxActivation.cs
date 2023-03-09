using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.HelperClasses;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    [ActivationType(Type = ActivationTypes.SoftMax)]
    internal class SoftMaxActivation : BaseActivation<SigmoidActivation>, IActivationFunction
    {
        public float MinValue => 0;

        public float MaxValue => 1;

        public float Activation(float value)
        {
            return 0;
            // return input.X / input.SumInputs;
        }

        public float Prime(float value)
        {
            return (1 - value) * value;
        }
    }


    public class Parameters
    {
        public float X { get; set; }
        public float SumInputs { get; set; }
    }
}
