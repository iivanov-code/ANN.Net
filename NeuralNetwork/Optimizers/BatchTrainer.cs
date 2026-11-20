using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using ANN.Net.Abstractions.Interfaces;
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

        /// <summary>
        /// Per-layer work that forgets what the replica recorded while propagating, for batches whose samples are
        /// propagated twice: once for their outputs and once more for their gradients
        /// </summary>
        public List<Action> Resetters { get; } = new List<Action>();

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
            options = Validate(samples, options);
            lock (padlock)
            {
                var used = Prepare(samples, options);
                var losses = new Quad[samples];
                Run(used, samples, options, (replica, i) => losses[i] = train(replica, i));
                Apply(used, samples);

                Quad total = 0;
                foreach (Quad loss in losses) total += loss;
                return total / samples;
            }
        }

        /// <summary>
        /// Training step on a loss of all outputs together. The samples are propagated on the replicas for their
        /// outputs, <paramref name="loss"/> gives the gradient with respect to every output, and every sample is
        /// propagated again and backpropagated with its gradient. The weights move by the sum of the gradients
        /// </summary>
        public Quad TrainOutputs(int samples, BatchOptions options, Func<TReplica, int, Quad[]> propagate, BatchLoss loss, Action<TReplica, int, Quad[]> backpropagate)
        {
            ArgumentNullException.ThrowIfNull(loss);
            options = Validate(samples, options);
            lock (padlock)
            {
                var used = Prepare(samples, options);
                var outputs = new Quad[samples][];
                Run(used, samples, options, (replica, i) => outputs[i] = (Quad[])propagate(replica, i).Clone());
                foreach (var (_, context) in used)
                {
                    foreach (Action reset in context.Resetters) reset();
                }

                var gradients = new Quad[samples][];
                Quad value = loss(outputs, gradients);
                for (int i = 0; i < samples; i++)
                {
                    if (gradients[i] == null) throw new InvalidOperationException($"The loss gave no gradient for sample {i}");
                }

                Run(used, samples, options, (replica, i) =>
                {
                    propagate(replica, i);
                    backpropagate(replica, i, gradients[i]);
                });
                Apply(used, 1);
                return value;
            }
        }

        private static BatchOptions Validate(int samples, BatchOptions options)
        {
            options ??= new BatchOptions();
            if (samples < 1) throw new ArgumentException("The batch needs at least one sample", nameof(samples));
            if (options.ChunkSize < 1) throw new ArgumentOutOfRangeException(nameof(options), options.ChunkSize, "The chunk size must be at least 1");
            if (options.MaxDegreeOfParallelism == 0 || options.MaxDegreeOfParallelism < -1)
                throw new ArgumentOutOfRangeException(nameof(options), options.MaxDegreeOfParallelism, "Must be -1 or at least 1");
            return options;
        }

        /// <summary>
        /// One replica per chunk, with nothing collected yet
        /// </summary>
        private List<(TReplica Replica, ReplicaContext Context)> Prepare(int samples, BatchOptions options)
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

            return used;
        }

        /// <summary>
        /// Every chunk on its own replica, its samples one after the other
        /// </summary>
        private static void Run(List<(TReplica Replica, ReplicaContext Context)> used, int samples, BatchOptions options, Action<TReplica, int> work)
        {
            Parallel.For(0, used.Count, new ParallelOptions { MaxDegreeOfParallelism = options.MaxDegreeOfParallelism }, c =>
            {
                int end = Math.Min(samples, (c + 1) * options.ChunkSize);
                for (int i = c * options.ChunkSize; i < end; i++) work(used[c].Replica, i);
            });
        }

        /// <summary>
        /// Updates every tensor once with the collected gradients, added up in chunk order and divided by
        /// <paramref name="divisor"/>
        /// </summary>
        private static void Apply(List<(TReplica Replica, ReplicaContext Context)> used, int divisor)
        {
            IReadOnlyList<ReplicaContext> contexts = used.Select(x => x.Context).ToList();
            for (int k = 0; k < contexts[0].Links.Count; k++)
            {
                var (master, _, values) = contexts[0].Links[k];
                master.ApplyAverage(values(), contexts.Select(x => x.Links[k].Collector).ToList(), divisor);
            }

            foreach (var finish in contexts[0].Finishers) finish(contexts);
        }
    }
}
