namespace ANN.Net.Abstractions.Models
{
    public class ErrorResult
    {
        public Quad Target { get; set; }
        public Quad Predicted { get; set; }
    }
}
