namespace ANN.Net.Abstractions.Settings
{
    /// <summary>
    /// How <c>TrainBatch</c> spreads a batch over the processors. The samples are split into chunks of
    /// <see cref="ChunkSize"/>; every chunk is computed on one thread, in sample order, and the gradients of the chunks
    /// are added in chunk order. The result therefore depends on the chunk size but not on the number of processors or
    /// threads, so the same training gives the same network on every machine
    /// </summary>
    public sealed class BatchOptions
    {
        /// <summary>
        /// Samples computed one after the other by one thread. Smaller chunks spread better over many processors;
        /// every chunk keeps one copy of the per-sample state and of the gradients, so larger chunks use less memory
        /// </summary>
        public int ChunkSize { get; set; } = 8;

        /// <summary>
        /// Most threads used at once; -1 uses every processor
        /// </summary>
        public int MaxDegreeOfParallelism { get; set; } = -1;
    }
}
