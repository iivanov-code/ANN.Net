using System.Collections.Generic;
using ANN.Net.Abstractions.Optimizers;

namespace ANN.Net.Abstractions.Interfaces
{
    /// <summary>
    /// Network that learns a dense vector for every token id of a vocabulary
    /// </summary>
    public interface IEmbeddingNetwork
    {
        int VocabularySize { get; }

        int EmbeddingDimension { get; }

        /// <summary>
        /// Live input embedding matrix, one row per token id
        /// </summary>
        Quad[][] Embeddings { get; }

        /// <summary>
        /// Scores of the last propagation, one per output token
        /// </summary>
        Quad[] Values { get; }

        /// <summary>
        /// Score minus target of the last backpropagation, one per output token
        /// </summary>
        Quad[] Errors { get; }

        ICollection<LearningRateOptimizer> Optimizers { get; }

        /// <summary>
        /// Scores every output token against the input tokens: the sigmoid of the dot product between
        /// the mean input embedding of the input tokens and the output embedding of the output token
        /// </summary>
        Quad[] Propagate(int[] inputTokens, int[] outputTokens);

        /// <summary>
        /// Binary cross entropy of the last propagated scores against the targets
        /// (1 for a real context token, 0 for a noise token). Updates both embedding
        /// matrices and returns the total loss
        /// </summary>
        Quad Backpropagate(Quad[] targets);

        /// <summary>
        /// One training pass over the token sequences: slides the context window,
        /// draws negative samples and backpropagates every sample. Returns the mean loss per sample
        /// </summary>
        Quad Train(IEnumerable<int[]> sequences);

        /// <summary>
        /// Copy of the input embedding of the token
        /// </summary>
        Quad[] GetEmbedding(int token);

        /// <summary>
        /// Copies of the input embeddings of the tokens, one vector per token
        /// </summary>
        Quad[][] Embed(int[] tokens);

        Quad CosineSimilarity(int first, int second);

        /// <summary>
        /// The tokens whose embeddings are closest to the given token, most similar first
        /// </summary>
        IReadOnlyList<(int Token, Quad Similarity)> MostSimilar(int token, int count);
    }
}
