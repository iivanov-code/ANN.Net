﻿using System.Collections;
using System.Collections.Generic;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.Abstractions.HelperClasses
{
    public class SynapseCollection<T> : ISynapseCollection<T>, ICollection<T>
        where T : ISynapse
    {
        private List<T> list;
        public SynapseCollection(ushort counter = 0)
        {
            ActivatedCount = counter;
            list = new List<T>();
        }

        public ushort ActivatedCount { get; private set; }

        public int Count => list.Count;
        public bool IsReadOnly => false;

        public void Add(T item)
        {
            ActivatedCount++;
            list.Add(item);
        }

        public void ResetCounter()
        {
            ActivatedCount = (ushort)list.Count;
        }

        public bool Zeroed()
        {
            return Count == ActivatedCount;
        }


        public void AccountSignal()
        {
            checked
            {
                ActivatedCount--;
            }
        }

        public void Clear()
        {
            list.Clear();
        }

        public bool Contains(T item)
        {
            return list.Contains(item);
        }

        public void CopyTo(T[] array, int arrayIndex)
        {
            list.CopyTo(array, arrayIndex);
        }

        public IEnumerator<T> GetEnumerator()
        {
            return list.GetEnumerator();
        }

        public bool Remove(T item)
        {
            return list.Remove(item);
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return list.GetEnumerator();
        }

        public bool CheckCountAndReset()
        {
            if (ActivatedCount == 0)
            {
                ActivatedCount = (ushort)Count;
                return true;
            }
            return false;
        }
    }
}