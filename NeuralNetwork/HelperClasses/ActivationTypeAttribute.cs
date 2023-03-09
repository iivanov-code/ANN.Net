using ANN.Net.Abstractions.Enums;
using System;

namespace ANN.Net.HelperClasses
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
    public sealed class ActivationTypeAttribute : Attribute
    {
        public ActivationTypes Type { get; set; }
        public ActivationTypeAttribute()
        { }
    }
}
