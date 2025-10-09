using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Models;

namespace ANN.Net.LossFunctions.ClassificationLosses
{
    internal class SoftMarginLossFunction : BaseLossFunction<SoftMarginLossFunction>, ILossFunction
    {
        public Quad CalculateLoss(IList<ErrorResult> result)
        {
            throw new NotImplementedException();
        }
    }
}
