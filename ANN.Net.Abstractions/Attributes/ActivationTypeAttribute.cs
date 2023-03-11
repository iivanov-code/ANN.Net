using System;
using ANN.Net.Abstractions.Enums;

namespace ANN.Net.Abstractions.Attributes
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
    public sealed class ActivationTypeAttribute : Attribute
    {
        public ActivationTypes Type { get; set; }

        public ActivationTypeAttribute()
        { }
    }
}
