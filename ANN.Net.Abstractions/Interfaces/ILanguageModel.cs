using System.Collections.Generic;
using ANN.Net.Abstractions.Settings;

namespace ANN.Net.Abstractions.Interfaces
{
    /// <summary>
    /// Tokenizer, optional token embeddings and a sequence network trained to predict the next token
    /// </summary>
    public interface ILanguageModel
    {
        ITokenizer Tokenizer { get; }

        ISequenceNetwork Network { get; }

        /// <summary>
        /// Embedding network that turns token ids into the input vectors; null when one-hot vectors are used
        /// </summary>
        IEmbeddingNetwork Embeddings { get; }

        /// <summary>
        /// Number of tokens the model sees at once
        /// </summary>
        int ContextLength { get; }

        /// <summary>
        /// Cuts the sequences into next-token windows and trains one epoch on them.
        /// Returns the mean cross entropy per predicted token
        /// </summary>
        Quad Train(IEnumerable<int[]> sequences);

        /// <summary>
        /// Mean cross entropy per predicted token over the sequences, without updating the weights
        /// </summary>
        Quad Evaluate(IEnumerable<int[]> sequences);

        /// <summary>
        /// Exponential of <see cref="Evaluate"/>; 1 is a perfect model, the vocabulary size is a uniform guess
        /// </summary>
        Quad Perplexity(IEnumerable<int[]> sequences);

        /// <summary>
        /// Probability distribution over the token that follows the prompt
        /// </summary>
        Quad[] PredictNext(int[] prompt);

        /// <summary>
        /// Continues the prompt. Returns only the new tokens, without the end token
        /// </summary>
        int[] Generate(int[] prompt, GenerationSettings settings);

        /// <summary>
        /// Encodes the prompt, continues it and decodes the continuation
        /// </summary>
        string Generate(string prompt, GenerationSettings settings);
    }
}
