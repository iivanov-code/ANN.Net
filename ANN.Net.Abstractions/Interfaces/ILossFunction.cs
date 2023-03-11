namespace ANN.Net.Abstractions.Interfaces
{
    public interface ILossFunction
    {
        Quad CalculateLoss(Quad target, Quad predicted);
    }
}
