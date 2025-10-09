using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Models;

namespace ANN.Net.LossFunctions.ContrastiveLosses
{
    internal class CosineEmbeddingLossFunction : BaseLossFunction<CosineEmbeddingLossFunction>, ILossFunction
    {
        public Quad CalculateLoss(IList<ErrorResult> result)
        {
            throw new NotImplementedException();
            //Cosine embedding loss function
            //https://pytorch.org/docs/stable/nn.html#cosineembeddingloss
        }
    }
}
