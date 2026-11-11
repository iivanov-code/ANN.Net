using System;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Optimizers;
using ANN.Net.Utils;
using Xunit;

namespace ANN.Net.Tests.Optimizers
{
    /// <summary>
    /// One <see cref="ArrayOptimizer"/> per array must change the values exactly like one optimizer per value
    /// </summary>
    public class ArrayOptimizerTests
    {
        [Theory]
        [MemberData(nameof(OptimizerTests.AllOptimizers), MemberType = typeof(OptimizerTests))]
        public void ArrayOptimizer_GivesBitForBitTheValuesOfOneOptimizerPerValue(OptimizerTypes type)
        {
            const int count = 17;
            var random = new Random((int)type + 1);
            var array = new ArrayOptimizer(type, count) { LearningRate = 0.03f };
            IOptimizer[] scalars = Enumerable.Range(0, count).Select(_ => NetworkUtils.GetOptimizer(type)).ToArray();
            foreach (LearningRateOptimizer scalar in scalars.Cast<LearningRateOptimizer>()) scalar.LearningRate = 0.03f;

            Quad[] arrayValues = Enumerable.Range(0, count).Select(_ => (Quad)((random.NextDouble() * 2) - 1)).ToArray();
            Quad[] scalarValues = (Quad[])arrayValues.Clone();

            for (int step = 0; step < 60; step++)
            {
                // zeros and sign changes, as real gradients have
                Quad[] gradients = Enumerable.Range(0, count).Select(i => (i + step) % 5 == 0 ? (Quad)0 : (Quad)((random.NextDouble() * 4) - 2)).ToArray();

                array.Update(arrayValues, gradients);
                for (int i = 0; i < count; i++) scalarValues[i] = scalars[i].Calculate(gradients[i], scalarValues[i]);

                Assert.Equal(scalarValues.Select(x => BitConverter.SingleToInt32Bits(x)), arrayValues.Select(x => BitConverter.SingleToInt32Bits(x)));
            }
        }

        [Theory]
        [InlineData(OptimizerTypes.Normal, 0)]
        [InlineData(OptimizerTypes.Momentum, 1)]
        [InlineData(OptimizerTypes.RMSprop, 1)]
        [InlineData(OptimizerTypes.Adam, 2)]
        [InlineData(OptimizerTypes.AmsGrad, 3)]
        public void ArrayOptimizer_KeepsOnlyTheStateOfItsRule(OptimizerTypes type, int state)
        {
            Assert.Equal(state, new ArrayOptimizer(type, 10).StatePerValue);
        }
    }
}
