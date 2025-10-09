using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Models;

namespace ANN.Net.LossFunctions.RankingLosses
{
    internal class MarginRankingLossFunction : BaseLossFunction<MarginRankingLossFunction>, ILossFunction
    {
        public Quad CalculateLoss(IList<ErrorResult> result)
        {
            throw new NotImplementedException();
        }
    }
}
