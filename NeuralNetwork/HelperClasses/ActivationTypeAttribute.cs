using NeuralNetwork.Enums;
using System;

namespace NeuralNetwork.HelperClasses
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
    public sealed class ActivationTypeAttribute : Attribute
    {
        public ActivationTypes Type { get; set; }
        public ActivationTypeAttribute()
        { }
    }
}
