using System;

[Serializable]
public struct Rate
{
    public static Rate MinValue = 0;
    public static Rate MaxValue = 1;

    private readonly float value;

    public Rate(float value)
    {
        if (value >= MinValue.value && value <= MaxValue.value)
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
        return this.value == ((Rate)obj).value;
    }

    public override int GetHashCode()
    {
        return this.value.GetHashCode();
    }

    public static bool operator ==(Rate left, Rate right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(Rate left, Rate right)
    {
        return !(left == right);
    }

    public static Rate operator -(Rate x, Rate y)
    {
        return x.value - y.value;
    }

    public static Rate operator -(Rate x, float y)
    {
        return x.value - y;
    }

    public static Rate operator +(Rate x, Rate y)
    {
        return x.value + y.value;
    }

    public static bool operator >(Rate x, Rate y)
    {
        return x.value > y.value;
    }

    public static bool operator <(Rate x, Rate y)
    {
        return x.value < y.value;
    }
}
