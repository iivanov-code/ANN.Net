using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    internal class HardTanHActivation : BaseActivation<HardTanHActivation>, IActivationFunction
    {
        public Quad MinValue => -1;

        public Quad MaxValue => 1;

        public Quad Activation(ref Quad x)
        {
            if (x > 1)
            {
                return 1;
            }
            else if (x < -1)
            {
                return -1;
            }
            else
            {
                return x;
            }
        }

        public Quad Derivative(Quad x)
        {
            return 1;
        }
    }
}
