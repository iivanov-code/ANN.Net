using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    internal class HardSwishActivation : BaseActivation<HardSwishActivation>, IActivationFunction
    {
        public Quad MinValue => -1;

        public Quad MaxValue => float.MaxValue;

        public Quad Activation(ref Quad x)
        {
            if (x <= -3) return 0;
            else if (x >= 3) return x;
            else
            {
                return x * (x + 3) / 6;
            }
        }

        public Quad Derivative(Quad x)
        {
            x += 3;

            if (x < 0)
            {
                return 0;
            }
            else
            {
                return 1;
            }
        }
    }
}
