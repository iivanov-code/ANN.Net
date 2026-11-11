using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using ANN.Net.Abstractions.Settings;

namespace ANN.Net.Optimizers
{
    /// <summary>
    /// Values of a network tensor, wherever the network keeps them
    /// </summary>
    internal delegate Span<Quad> TensorValues();

    /// <summary>
    /// Built while a training replica is made: every trainable tensor of the network is shared with the replica, and
    /// the replica gets a collector in place of the tensor's optimizer
    /// </summary>
    internal sealed class ReplicaContext
    {
        private readonly List<(ArrayOptimizer Master, ArrayOptimizer Collector, TensorValues Values)> links = new List<(ArrayOptimizer, ArrayOptimizer, TensorValues)>();

        public IReadOnlyList<(ArrayOptimizer Master, ArrayOptimizer Collector, TensorValues Values)> Links => links;

        /// <summary>
        /// Per-layer work to finish a batch, such as the running averages of batch normalization
        /// </summary>
        public List<Action<IReadOnlyList<ReplicaContext>>> Finishers { get; } = new List<Action<IReadOnlyList<ReplicaContext>>>();

        public ArrayOptimizer Link(ArrayOptimizer master, Quad[] values) => Link(master, () => values);

        public ArrayOptimizer Link(ArrayOptimizer master, Quad[,] values) =>
            Link(master, () => MemoryMarshal.CreateSpan(ref Unsafe.As<byte, Quad>(ref MemoryMarshal.GetArrayDataReference(values)), values.Length));

        private ArrayOptimizer Link(ArrayOptimizer master, TensorValues values)
        {
            ArrayOptimizer collector = ArrayOptimizer.Collector(master);
            links.Add((master, collector, values));
            return collector;
        }

        /// <summary>
        /// State the layers keep per replica, by the index of the layer in the order the replica was built
        /// </summary>
        public List<object> LayerState { get; } = new List<object>();
    }

    /// <summary>
    /// Mini-batch training on several threads. The batch is split into chunks of a fixed size; every chunk runs on a
    /// replica of the network that shares its weights, computing its samples one after the other. The weights do not
    /// change during the batch; afterwards every tensor is updated once with the mean gradient, summed in chunk order.
    /// The result therefore does not depend on the number of threads
    /// </summary>
    internal sealed class BatchTrainer<TReplica>
    {
        private readonly Func<ReplicaContext, TReplica> createReplica;
        private readonly List<(TReplica Replica, ReplicaContext Context)> replicas = new List<(TReplica, ReplicaContext)>();
        private readonly object padlock = new object();

        public BatchTrainer(Func<ReplicaContext, TReplica> createReplica)
        {
            this.createReplica = createReplica;
        }

        /// <param name="train">Trains the replica on sample i and returns its loss</param>
        public Quad Train(int samples, BatchOptions options, Func<TReplica, int, Quad> train)
        {
            options ??= new BatchOptions();
            if (samples < 1) throw new ArgumentException("The batch needs at least one sample", nameof(samples));
            if (options.ChunkSize < 1) throw new ArgumentOutOfRangeException(nameof(options), options.ChunkSize, "The chunk size must be at least 1");
            if (options.MaxDegreeOfParallelism == 0 || options.MaxDegreeOfParallelism < -1)
                throw new ArgumentOutOfRangeException(nameof(options), options.MaxDegreeOfParallelism, "Must be -1 or at least 1");

            lock (padlock)
            {
                int chunks = (samples + options.ChunkSize - 1) / options.ChunkSize;
                while (replicas.Count < chunks)
                {
                    var context = new ReplicaContext();
                    replicas.Add((createReplica(context), context));
                }

                var used = replicas.Take(chunks).ToList();
                foreach (var (_, context) in used)
                {
                    foreach (var link in context.Links) link.Collector.ClearCollected();
                }

                var losses = new Quad[samples];
                Parallel.For(0, chunks, new ParallelOptions { MaxDegreeOfParallelism = options.MaxDegreeOfParallelism }, c =>
                {
                    int end = Math.Min(samples, (c + 1) * options.ChunkSize);
                    for (int i = c * options.ChunkSize; i < end; i++) losses[i] = train(used[c].Replica, i);
                });

                IReadOnlyList<ReplicaContext> contexts = used.Select(x => x.Context).ToList();
                for (int k = 0; k < contexts[0].Links.Count; k++)
                {
                    var (master, _, values) = contexts[0].Links[k];
                    master.ApplyAverage(values(), contexts.Select(x => x.Links[k].Collector).ToList(), samples);
                }

                foreach (var finish in contexts[0].Finishers) finish(contexts);

                Quad total = 0;
                foreach (Quad loss in losses) total += loss;
                return total / samples;
            }
        }
    }
}
