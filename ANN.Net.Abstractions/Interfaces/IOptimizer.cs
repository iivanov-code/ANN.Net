namespace ANN.Net.Abstractions.Interfaces
{
    public interface IOptimizer
    {
        Quad Calculate(Quad gradient, Quad weight);
    }
}
