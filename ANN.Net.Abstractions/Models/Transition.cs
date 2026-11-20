namespace ANN.Net.Abstractions.Models
{
    /// <summary>
    /// One step of an agent, for Q-learning: in <see cref="State"/> it took <see cref="Action"/>, received
    /// <see cref="Reward"/> and ended up in <see cref="NextState"/>
    /// </summary>
    public sealed class Transition
    {
        public Quad[] State { get; set; }

        /// <summary>
        /// Index of the network output that belongs to the action
        /// </summary>
        public int Action { get; set; }

        public Quad Reward { get; set; }

        /// <summary>
        /// State after the action, or null when the episode ended with it, so nothing more is to be expected
        /// </summary>
        public Quad[] NextState { get; set; }
    }
}
