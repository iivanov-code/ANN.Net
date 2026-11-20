using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Models;
using ANN.Net.Abstractions.Settings;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Q-learning loss: the network gives the value Q(s, a) of every action a in state s, one output per action. For a
    /// batch of transitions the loss is the mean of (Q(s, a) - (r + γ max Q(s', a')))², where the target
    /// r + γ max Q(s', a') is computed before the step and is just r when the episode ended
    /// </summary>
    public sealed class QValueLoss
    {
        /// <param name="discount">γ, between 0 and 1: how much a reward one step later is worth now</param>
        public QValueLoss(float discount)
        {
            if (!(discount >= 0 && discount <= 1)) throw new ArgumentOutOfRangeException(nameof(discount), discount, "Must be between 0 and 1");
            Discount = discount;
        }

        public float Discount { get; }

        /// <summary>
        /// r + γ max Q(s', a') of every transition, with the values of <paramref name="network"/>; r alone when the
        /// episode ended
        /// </summary>
        public Quad[] Targets(IVectorNetwork network, IReadOnlyList<Transition> transitions)
        {
            ArgumentNullException.ThrowIfNull(network);
            ArgumentNullException.ThrowIfNull(transitions);

            var targets = new Quad[transitions.Count];
            for (int i = 0; i < targets.Length; i++)
            {
                Transition transition = transitions[i] ?? throw new ArgumentException("Every transition must be given", nameof(transitions));
                double target = transition.Reward;
                if (transition.NextState != null) target += Discount * (double)network.Propagate(transition.NextState).Max();
                targets[i] = (Quad)target;
            }

            return targets;
        }

        /// <param name="values">Q values of every action, per sample</param>
        /// <param name="actions">Action taken in every sample</param>
        /// <param name="targets">r + γ max Q(s', a') of every sample</param>
        public Quad CalculateLoss(IReadOnlyList<Quad[]> values, IReadOnlyList<int> actions, IReadOnlyList<Quad> targets) => Calculate(values, actions, targets, null);

        /// <summary>
        /// Gradient of <see cref="CalculateLoss"/> with respect to every Q value: 0 except for the action taken
        /// </summary>
        public Quad[][] CalculateGradient(IReadOnlyList<Quad[]> values, IReadOnlyList<int> actions, IReadOnlyList<Quad> targets)
        {
            BatchLossChecks.SameCount(values, actions, nameof(values), nameof(actions));
            BatchLossChecks.Vectors(values, nameof(values));
            Quad[][] gradients = values.Select(x => new Quad[x.Length]).ToArray();
            Calculate(values, actions, targets, gradients);
            return gradients;
        }

        /// <summary>
        /// One Q-learning step on a batch of transitions. The targets come from <paramref name="targetNetwork"/>, a copy
        /// of the network that is updated less often to keep them steady, or from the network itself when it is null.
        /// See <see cref="IVectorNetwork.TrainOutputs"/>; returns the loss before the step
        /// </summary>
        public Quad Train(IVectorNetwork network, IReadOnlyList<Transition> transitions, IVectorNetwork targetNetwork = null, BatchOptions options = null)
        {
            ArgumentNullException.ThrowIfNull(network);
            Quad[] targets = Targets(targetNetwork ?? network, transitions);
            int[] actions = transitions.Select(x => x.Action).ToArray();
            List<Quad[]> states = transitions.Select(x => x.State).ToList();
            BatchLossChecks.Vectors(states, nameof(transitions));

            return network.TrainOutputs(states, (outputs, gradients) =>
            {
                for (int i = 0; i < outputs.Count; i++) gradients[i] = new Quad[outputs[i].Length];
                return Calculate(outputs, actions, targets, gradients);
            }, options);
        }

        private static Quad Calculate(IReadOnlyList<Quad[]> values, IReadOnlyList<int> actions, IReadOnlyList<Quad> targets, Quad[][] gradients)
        {
            int n = BatchLossChecks.SameCount(values, actions, nameof(values), nameof(actions));
            BatchLossChecks.SameCount(values, targets, nameof(values), nameof(targets));
            BatchLossChecks.Actions(values, actions, nameof(actions));
            if (n == 0) return 0;

            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                double error = (double)values[i][actions[i]] - targets[i];
                sum += error * error;
                if (gradients != null) gradients[i][actions[i]] = (Quad)(2 * error / n);
            }

            return (Quad)(sum / n);
        }
    }
}
