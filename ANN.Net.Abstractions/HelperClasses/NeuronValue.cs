using System;

namespace ANN.Net.Abstractions.HelperClasses
{
    public struct NeuronValue : IEquatable<NeuronValue>
    {
        private float[] value;

        public NeuronValue(float value)
        {
            this.value = new float[1] { value };
        }

        public NeuronValue(float[] values)
        {
            value = values;
        }

        public int Length
        {
            get
            {
                return value.Length;
            }
        }

        public float this[int index]
        {
            get
            {
                return value[index];
            }
        }

        public static implicit operator float(NeuronValue value)
        {
            return value.value[0];
        }

        public static implicit operator float[](NeuronValue value)
        {
            return value.value;
        }

        public static implicit operator NeuronValue(float value)
        {
            return new NeuronValue(value);
        }

        public static implicit operator NeuronValue(float[] value)
        {
            return new NeuronValue(value);
        }

        public override bool Equals(object obj)
        {
            return Equals((NeuronValue)obj);
        }

        public bool Equals(NeuronValue other)
        {
            for (int i = 0; i < other.value.Length; i++)
            {
                if (value[i] != other.value[i])
                {
                    return false;
                }
            }
            return true;
        }

        public override int GetHashCode()
        {
            return 0;
        }
    }
}
