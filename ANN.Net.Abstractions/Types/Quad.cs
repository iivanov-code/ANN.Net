using System;
using System.Runtime.Serialization;

[Serializable]
public struct Quad : IEquatable<Quad>, IComparable<Quad>, ISerializable, IConvertible
{
    public static readonly Quad MaxValue = new Quad(float.MaxValue);
    public static readonly Quad MinValue = new Quad(float.MinValue);
    public static readonly Quad NegativeInfinity = new Quad(float.NegativeInfinity);
    public static readonly Quad PositiveInfinity = new Quad(float.PositiveInfinity);
    public static readonly Quad Zero = new Quad(0);

    private float value;

    public Quad(float value)
    {
        this.value = value;
    }

    public static explicit operator Quad(double value)
    {
        return new Quad((float)value);
    }

    public static implicit operator float(Quad value)
    {
        return value;
    }

    public static implicit operator Quad(byte value)
    {
        return new Quad(value);
    }

    public static implicit operator Quad(int value)
    {
        return new Quad(value);
    }

    public static implicit operator Quad(float value)
    {
        return new Quad(value);
    }

    public static Quad operator -(Quad x)
    {
        return -x.value;
    }

    public static Quad operator -(Quad x, Quad y)
    {
        return x.value - y.value;
    }

    public static bool operator !=(Quad x, Quad y)
    {
        return x.value != y.value;
    }

    public static Quad operator *(Quad x, Quad y)
    {
        return x.value * y.value;
    }

    public static Quad operator /(Quad x, Quad y)
    {
        return x.value / y.value;
    }

    public static Quad operator +(Quad x)
    {
        return x;
    }

    public static Quad operator +(Quad x, Quad y)
    {
        return x.value + y.value;
    }

    public static bool operator <(Quad x, Quad y)
    {
        return x.value < y.value;
    }

    public static bool operator <=(Quad x, Quad y)
    {
        return x.value <= y.value;
    }

    public static bool operator ==(Quad x, Quad y)
    {
        return x.value == y.value;
    }

    public static bool operator >(Quad x, Quad y)
    {
        return x.value > y.value;
    }

    public static bool operator >=(Quad x, Quad y)
    {
        return x.value >= y.value;
    }

    public int CompareTo(Quad other)
    {
        return this.value.CompareTo(other.value);
    }

    public override bool Equals(object obj)
    {
        if (!(obj is Quad)) return false;
        Quad x = (Quad)obj;
        return value == x.value;
    }

    public bool Equals(Quad other)
    {
        return this.value == other.value;
    }

    public override int GetHashCode()
    {
        return value.GetHashCode();
    }

    public void GetObjectData(SerializationInfo info, StreamingContext context)
    {
        throw new NotImplementedException();
    }

    public TypeCode GetTypeCode()
    {
        return TypeCode.Double;
    }

    public bool ToBoolean(IFormatProvider provider)
    {
        return value == 0;
    }

    public byte ToByte(IFormatProvider provider)
    {
        return (byte)value;
    }

    public char ToChar(IFormatProvider provider)
    {
        return char.Parse(char.ConvertFromUtf32((int)value));
    }

    public DateTime ToDateTime(IFormatProvider provider)
    {
        throw new NotImplementedException();
    }

    public decimal ToDecimal(IFormatProvider provider)
    {
        return ((decimal)value);
    }

    public double ToDouble(IFormatProvider provider)
    {
        return value;
    }

    public short ToInt16(IFormatProvider provider)
    {
        return (short)value;
    }

    public int ToInt32(IFormatProvider provider)
    {
        return (int)value;
    }

    public long ToInt64(IFormatProvider provider)
    {
        return (long)value;
    }

    public sbyte ToSByte(IFormatProvider provider)
    {
        return (sbyte)value;
    }

    public float ToSingle(IFormatProvider provider)
    {
        return value;
    }

    public override string ToString()
    {
        return value.ToString();
    }

    public string ToString(IFormatProvider provider)
    {
        return value.ToString(provider);
    }

    public object ToType(Type conversionType, IFormatProvider provider)
    {
        return Convert.ChangeType(value, conversionType);
    }

    public ushort ToUInt16(IFormatProvider provider)
    {
        return ((ushort)value);
    }

    public uint ToUInt32(IFormatProvider provider)
    {
        return (uint)value;
    }

    public ulong ToUInt64(IFormatProvider provider)
    {
        return (ulong)value;
    }
}
