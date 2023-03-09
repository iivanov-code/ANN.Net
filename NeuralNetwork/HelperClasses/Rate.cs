using System;

namespace ANN.Net.HelperClasses
{
    [Serializable]
    public struct Rate
    {
        private readonly float value;

        public Rate(float value)
        {
            if (value >= 0 && value <= 1)
            {
                this.value = value;
            }
            else
            {
                throw new ArgumentException("Invalid value");
            }
        }

        public static implicit operator Rate(float value)
        {
            return new Rate(value);
        }

        public static implicit operator float(Rate value)
        {
            return value.value;
        }

        public override bool Equals(object obj)
        {
            return value == ((Rate)obj).value;
        }

        public override int GetHashCode()
        {
            return value.GetHashCode();
        }

        public static bool operator ==(Rate left, Rate right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Rate left, Rate right)
        {
            return !(left == right);
        }
    }
}
