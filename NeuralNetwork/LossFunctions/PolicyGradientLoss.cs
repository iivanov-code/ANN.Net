using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Settings;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Policy gradient (REINFORCE) loss: the network gives the probability π(a | s) of every action a in state s, one
    /// output per action, so its outputs should be a probability distribution (a soft-max network). For a batch of
    /// steps the loss is -mean(Q · ln π(a | s)), with Q the return of the step or its advantage: actions that turned
    /// out well become more likely, the others less likely
    /// </summary>
    public sealed class PolicyGradientLoss
    {
        private const double Epsilon = 1e-7;

        /// <param name="probabilities">Probability of every action, per step</param>
        /// <param name="actions">Action taken in every step</param>
        /// <param name="returns">Q, how well every step turned out</param>
        public Quad CalculateLoss(IReadOnlyList<Quad[]> probabilities, IReadOnlyList<int> actions, IReadOnlyList<Quad> returns) =>
            Calculate(probabilities, actions, returns, null);

        /// <summary>
        /// Gradient of <see cref="CalculateLoss"/> with respect to every probability: 0 except for the action taken
        /// </summary>
        public Quad[][] CalculateGradient(IReadOnlyList<Quad[]> probabilities, IReadOnlyList<int> actions, IReadOnlyList<Quad> returns)
        {
            BatchLossChecks.SameCount(probabilities, actions, nameof(probabilities), nameof(actions));
            BatchLossChecks.Vectors(probabilities, nameof(probabilities));
            Quad[][] gradients = probabilities.Select(x => new Quad[x.Length]).ToArray();
            Calculate(probabilities, actions, returns, gradients);
            return gradients;
        }

        /// <summary>
        /// One policy gradient step on a batch of steps; see <see cref="IVectorNetwork.TrainOutputs"/>. Returns the loss
        /// before the step
        /// </summary>
        public Quad Train(IVectorNetwork network, IReadOnlyList<Quad[]> states, IReadOnlyList<int> actions, IReadOnlyList<Quad> returns, BatchOptions options = null)
        {
            ArgumentNullException.ThrowIfNull(network);
            BatchLossChecks.SameCount(states, actions, nameof(states), nameof(actions));
            BatchLossChecks.SameCount(states, returns, nameof(states), nameof(returns));

            return network.TrainOutputs(states, (outputs, gradients) =>
            {
                for (int i = 0; i < outputs.Count; i++) gradients[i] = new Quad[outputs[i].Length];
                return Calculate(outputs, actions, returns, gradients);
            }, options);
        }

        private static Quad Calculate(IReadOnlyList<Quad[]> probabilities, IReadOnlyList<int> actions, IReadOnlyList<Quad> returns, Quad[][] gradients)
        {
            int n = BatchLossChecks.SameCount(probabilities, actions, nameof(probabilities), nameof(actions));
            BatchLossChecks.SameCount(probabilities, returns, nameof(probabilities), nameof(returns));
            BatchLossChecks.Actions(probabilities, actions, nameof(actions));
            if (n == 0) return 0;

            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                double probability = Math.Max(probabilities[i][actions[i]], Epsilon);
                sum -= returns[i] * Math.Log(probability);
                if (gradients != null) gradients[i][actions[i]] = (Quad)(-returns[i] / (probability * n));
            }

            return (Quad)(sum / n);
        }
    }
}
