using System.Collections.Generic;
using ANN.Net.Abstractions.Models;

namespace ANN.Net.Abstractions.Interfaces
{
    public interface ILossFunction
    {
        Quad CalculateLoss(IList<ErrorResult> result);
    }
}
