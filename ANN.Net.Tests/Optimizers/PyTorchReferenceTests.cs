using System;
using System.Collections.Generic;
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
    /// Every optimizer against the weights PyTorch's own optimizers produce, so a rule cannot pass by agreeing with a
    /// hand-written copy of itself
    /// </summary>
    public class PyTorchReferenceTests
    {
        private const float LearningRate = 0.1f;
        private const float StartWeight = 0.7f;

        private static readonly float[] Gradients = { 0, 0.5f, -1.2f, 2, 0, -0.3f, 0.7f, 1e-3f, -4, 0.25f, 0, 0, 1.5f };

        /// <summary>
        /// The weight after every step, from PyTorch 2.15 in float64 with lr = 0.1, a start weight of 0.7 and the gradients
        /// above, set by hand before each <c>step()</c>:
        /// <code>
        /// Normal   SGD(lr)                          Momentum  SGD(lr, momentum=0.9)
        /// Nesterov SGD(lr, momentum=0.9, nesterov=True)
        /// Adam     Adam(lr, betas=(0.9, 0.999), eps=1e-8)
        /// AdaMax   Adamax(lr, betas=(0.9, 0.999), eps=1e-8)
        /// Nadam    NAdam(lr, betas=(0.9, 0.999), eps=1e-8, momentum_decay=1e6)
        /// AmsGrad  Adam(lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=True)
        /// RMSprop  RMSprop(lr, alpha=0.9, eps=1e-8)
        /// </code>
        /// PyTorch's NAdam uses the momentum mu_t = beta_1 * (1 - 0.5 * 0.96^(t * momentum_decay)); a huge momentum_decay
        /// makes it the constant beta_1 of ANN.Net's Nadam (momentum_decay=0 would make it beta_1 / 2)
        /// </summary>
        private static readonly Dictionary<OptimizerTypes, double[]> Expected = new Dictionary<OptimizerTypes, double[]>
        {
            [OptimizerTypes.Normal] = new[] { 0.700000000, 0.650000000, 0.770000000, 0.570000000, 0.570000000, 0.600000000, 0.530000000, 0.529900000, 0.929900000, 0.904900000, 0.904900000, 0.904900000, 0.754900000 },
            [OptimizerTypes.Momentum] = new[] { 0.700000000, 0.650000000, 0.725000000, 0.592500000, 0.473250000, 0.395925000, 0.256332500, 0.130599250, 0.417439325, 0.650595392, 0.860435853, 1.049292268, 1.069263041 },
            [OptimizerTypes.Nesterov] = new[] { 0.700000000, 0.605000000, 0.792500000, 0.473250000, 0.365925000, 0.326332500, 0.130699250, 0.017439325, 0.675595393, 0.860435853, 1.049292268, 1.219263041, 1.087236737 },
            [OptimizerTypes.Adam] = new[] { 0.700000000, 0.625586320, 0.662443676, 0.630158300, 0.602869795, 0.586057572, 0.557785068, 0.532841320, 0.562570181, 0.586499463, 0.607944599, 0.627226679, 0.629172886 },
            [OptimizerTypes.AdaMax] = new[] { 0.700000000, 0.647368422, 0.670431153, 0.651166832, 0.636592173, 0.628324283, 0.614905529, 0.603822985, 0.615529224, 0.624487527, 0.632147977, 0.638747472, 0.639419586 },
            [OptimizerTypes.Nadam] = new[] { 0.700000000, 0.578631593, 0.663743232, 0.590609133, 0.569144552, 0.562077581, 0.524591721, 0.503700057, 0.570321573, 0.588197782, 0.606654340, 0.623351042, 0.610426985 },
            [OptimizerTypes.AmsGrad] = new[] { 0.700000000, 0.625586320, 0.662443676, 0.630158300, 0.602883443, 0.586071220, 0.557798716, 0.532867440, 0.562596301, 0.586525584, 0.607959995, 0.627222792, 0.629169000 },
            [OptimizerTypes.RMSprop] = new[] { 0.700000000, 0.383772254, 0.677858096, 0.408141372, 0.408141372, 0.452646737, 0.349205564, 0.349049797, 0.633970605, 0.615232780, 0.615232780, 0.615232780, 0.493674021 },
        };

        [Fact]
        public void EveryOptimizerType_HasAPyTorchReference()
        {
            Assert.Equal(Enum.GetValues<OptimizerTypes>().OrderBy(x => x), Expected.Keys.OrderBy(x => x));
        }

        [Theory]
        [MemberData(nameof(OptimizerTests.AllOptimizers), MemberType = typeof(OptimizerTests))]
        public void Optimizer_GivesTheWeightsOfPyTorch(OptimizerTypes type)
        {
            IOptimizer optimizer = NetworkUtils.GetOptimizer(type);
            ((LearningRateOptimizer)optimizer).LearningRate = LearningRate;
            Quad weight = StartWeight;

            for (int step = 0; step < Gradients.Length; step++)
            {
                weight = optimizer.Calculate(Gradients[step], weight);
                AssertClose(type, step, Expected[type][step], weight);
            }
        }

        [Theory]
        [MemberData(nameof(OptimizerTests.AllOptimizers), MemberType = typeof(OptimizerTests))]
        public void ArrayOptimizer_GivesTheWeightsOfPyTorch(OptimizerTypes type)
        {
            var optimizer = new ArrayOptimizer(type, 1) { LearningRate = LearningRate };
            Quad[] weight = { StartWeight };

            for (int step = 0; step < Gradients.Length; step++)
            {
                optimizer.Update(weight, new Quad[] { Gradients[step] });
                AssertClose(type, step, Expected[type][step], weight[0]);
            }
        }

        private static void AssertClose(OptimizerTypes type, int step, double expected, Quad actual)
        {
            // float32 against float64: a few units in the last place of float, accumulated over the steps
            Assert.True(Math.Abs(expected - actual) <= 1e-5 * Math.Max(1, Math.Abs(expected)),
                        $"{type} step {step}: PyTorch {expected:R}, ANN.Net {(float)actual:R}");
        }

        private static float[] Steps(OptimizerTypes type, IEnumerable<float> gradients)
        {
            IOptimizer optimizer = NetworkUtils.GetOptimizer(type);
            ((LearningRateOptimizer)optimizer).LearningRate = LearningRate;
            Quad weight = 0;
            var steps = new List<float>();
            foreach (float gradient in gradients)
            {
                Quad next = optimizer.Calculate(gradient, weight);
                steps.Add(weight - next);
                weight = next;
            }

            return steps.ToArray();
        }

        /// <summary>
        /// With bias correction the adaptive optimizers start with steps of about the learning rate. Without it AMSGrad
        /// took steps of up to 6.6 times the rate (around step 12) under a constant gradient
        /// </summary>
        [Theory]
        [InlineData(OptimizerTypes.Adam, 1.0)]
        [InlineData(OptimizerTypes.AdaMax, 1.0)]
        [InlineData(OptimizerTypes.AmsGrad, 1.0)]
        [InlineData(OptimizerTypes.Nadam, 1.48)] // Nesterov look-ahead: 1.474 at the first step in PyTorch too
        public void AdaptiveOptimizer_StepsAreAtMostAboutTheLearningRate(OptimizerTypes type, double largest)
        {
            float[] steps = Steps(type, Enumerable.Repeat(1f, 200));

            Assert.All(steps, step => Assert.InRange(step, 0, (largest * LearningRate) + 1e-6));
            Assert.Equal(LearningRate, steps[^1], 4);
        }

        [Fact]
        public void AmsGrad_StepsExactlyLikeAdam_WhileTheSquaredAverageGrows()
        {
            float[] gradients = Enumerable.Range(1, 40).Select(i => 0.1f * i).ToArray();

            Assert.Equal(Steps(OptimizerTypes.Adam, gradients), Steps(OptimizerTypes.AmsGrad, gradients));
        }

        [Fact]
        public void AmsGrad_StepsLessThanAdam_AfterTheGradientsShrink()
        {
            float[] gradients = new[] { 4f, 4f, 4f }.Concat(Enumerable.Repeat(0.01f, 3000)).ToArray();
            float[] adam = Steps(OptimizerTypes.Adam, gradients), amsGrad = Steps(OptimizerTypes.AmsGrad, gradients);

            for (int i = 0; i < gradients.Length; i++)
            {
                Assert.True(Math.Abs(amsGrad[i]) <= Math.Abs(adam[i]) + 1e-7, $"step {i}: AmsGrad {amsGrad[i]}, Adam {adam[i]}");
            }

            // the largest squared average so far keeps AMSGrad's steps small, while Adam's average forgets the large
            // gradients (about 0.23 of Adam's step after 3000 small ones)
            Assert.True(amsGrad[^1] < 0.5 * adam[^1], $"AmsGrad {amsGrad[^1]}, Adam {adam[^1]}");
        }
    }
}
