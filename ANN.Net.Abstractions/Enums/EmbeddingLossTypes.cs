namespace ANN.Net.Abstractions.Enums
{
    /// <summary>
    /// Losses of pairs of embeddings x1 and x2 with label y: 1 when they should be alike, -1 when they should differ
    /// </summary>
    public enum EmbeddingLossTypes
    {
        /// <summary>
        /// 1 - cos(x1, x2) for y = 1, max(0, cos(x1, x2) - α) for y = -1, with the margin α between -1 and 1
        /// </summary>
        CosineEmbedding,

        /// <summary>
        /// d for y = 1, max(0, δ - d) for y = -1, with d = Σ(x1 - x2)², the squared distance, and the margin δ
        /// </summary>
        HingeEmbedding,
    }
}
