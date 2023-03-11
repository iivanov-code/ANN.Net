using System.Collections;
using System.Collections.Generic;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Synapses;

namespace ANN.Net.Abstractions.Collections
{
    public class SynapseCollection<T> : ISynapseCollection<T>, IEnumerable<T>
         where T : ISynapse
    {
        protected uint activatedCount;
        protected uint recurrentCount = 0;
        protected List<T> list;
        public Quad Value { get; set; } = 0;

        public SynapseCollection()
        {
            list = new List<T>();
        }

        public void Add(T item)
        {
            list.Add(item);
        }

        public void AddRecurrent(T axon)
        {
            recurrentCount++;
            list.Add(axon);
        }

        public void ClearValue()
        {
            Value = 0;
        }

        public IEnumerator<T> GetEnumerator()
        {
            return list.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return list.GetEnumerator();
        }

        public bool Remove(T item)
        {
            return list.Remove(item);
        }
    }
}
