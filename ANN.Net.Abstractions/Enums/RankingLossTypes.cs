namespace ANN.Net.Abstractions.Enums
{
    /// <summary>
    /// Losses over all ordered pairs (i, j) of a batch of N scores ŷ with relevances y, divided by N². With
    /// s = 1 - 2·[y_i &gt; y_j], which is -1 when sample i should score higher than sample j and 1 otherwise, and margin δ:
    /// </summary>
    public enum RankingLossTypes
    {
        /// <summary>
        /// max(0, s (ŷ_i - ŷ_j) + δ)
        /// </summary>
        MarginRanking,

        /// <summary>
        /// ln(1 + exp(s (ŷ_i - ŷ_j) + δ)), a smooth <see cref="MarginRanking"/>
        /// </summary>
        SoftPairwiseHinge,

        /// <summary>
        /// ln(1 + exp(s (ŷ_i - ŷ_j))), without a margin
        /// </summary>
        PairwiseLogistic,
    }
}
