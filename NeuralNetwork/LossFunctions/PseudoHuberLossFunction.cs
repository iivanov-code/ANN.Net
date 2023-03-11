using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.LossFunctions
{
    internal class PseudoHuberLossFunction : ILossFunction
    {
        private Quad alpha;

        public PseudoHuberLossFunction(float alpha)
        {
            this.alpha = alpha;
        }

        public Quad CalculateLoss(Quad target, Quad predicted)
        {
            Quad absolute = AbsoluteLossFunction.Instance.CalculateLoss(target, predicted);

            if (absolute <= alpha)
            {
                return SquaredLossFunction.Instance.CalculateLoss(target, predicted);
            }
            else
            {
                return absolute;
            }
        }
    }
}
