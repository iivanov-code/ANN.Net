using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;

namespace ANN.Net.Abstractions.Collections
{
    [Serializable]
    public class Layer<T> : ILayer<T>, IReadOnlyCollection<T>, IEnumerable<T>, ICollection, IEnumerable
        where T : class, INeuron
    {
        public Layer()
        {
            this.neurons = new List<INeuron>();
        }

        public Layer(IEnumerable<T> list)
        {
            this.neurons = new List<INeuron>(list);
        }

        public Layer(int count)
        {
            this.neurons = new List<INeuron>(count);
        }

        private List<INeuron> neurons;

        public int Count => neurons.Count;

        public bool IsSynchronized => false;

        public object SyncRoot => null;

        public T this[int index]
        {
            get
            {
                return neurons[index] as T;
            }
        }

        public void AddRange(IEnumerable<T> range)
        {
            this.neurons.AddRange(range);
        }

        public IEnumerator<T> GetEnumerator()
        {
            return this.neurons.Cast<T>().GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }

        public void Add(T neuron)
        {
            this.neurons.Add(neuron);
        }

        public ILayer<TNeuron> AsLayerOf<TNeuron>()
             where TNeuron : class, INeuron
        {
            return new Layer<TNeuron>(this.neurons.Cast<TNeuron>());
        }

        public void CopyTo(Array array, int index)
        {
        }
    }
}
