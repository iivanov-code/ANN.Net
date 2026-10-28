using System.Collections.Generic;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Models;

namespace ANN.Net.LossFunctions
{
    internal class DifferenceLossFunction : BaseLossFunction<DifferenceLossFunction>, ILossFunction
    {
        public Quad CalculateLoss(IList<ErrorResult> result)
        {
            Quad sum = 0;
            foreach (var item in result)
            {
                sum += item.Predicted - item.Target;
            }

            return sum;
        }
    }
}
