namespace ANN.Net.Abstractions.Enums
{
    public enum Word2VecArchitecture
    {
        /// <summary>
        /// Predicts every context token from the center token
        /// </summary>
        SkipGram = 0,

        /// <summary>
        /// Predicts the center token from the mean of its context tokens
        /// </summary>
        ContinuousBagOfWords = 1
    }
}
