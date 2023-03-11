using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.LossFunctions
{
    internal class DifferenceLossFunction : BaseLossFunction<DifferenceLossFunction>, ILossFunction
    {
        public Quad CalculateLoss(Quad target, Quad predicted)
        {
            return predicted - target;
        }
    }
}
