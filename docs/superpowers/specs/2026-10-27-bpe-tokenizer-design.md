# BPE tokenizer design

Date: 2026-10-27

## Goal

Add a byte-pair-encoding (BPE) tokenizer to ANN.Net so text can be turned into
token ids for the transformer (and back), behind an interface that lets the
caller swap in a different tokenizer later.

## Decisions

- **Byte-level BPE.** The base vocabulary is the 256 byte values, so any UTF-8
  text (Latin, Cyrillic, emoji) encodes without an unknown token and
  `Decode(Encode(text)) == text` always holds, trained or not.
- **Chunks bound merges.** Text is pre-split into chunks: a run of non-whitespace
  characters together with the whitespace that precedes it (`"Hello"`, `" world"`,
  `"\n\n"` at the end). Merges never cross a chunk boundary, which keeps tokens
  word-like and stops training from learning cross-word pairs.
- **Special tokens** (pad, begin, end, ...) are configured by name, get ids right
  after the 256 bytes and before any merge, are never produced by `Encode` from
  plain text, and decode back to their name. Their ids are exposed through
  `ITokenizer.SpecialTokens`.
- **Deterministic training.** The most frequent adjacent pair is merged first;
  ties break on the lowest (left, right) id pair. Training stops at the target
  vocabulary size or when no pair reaches `MinimumPairFrequency`.
- **Library conventions.** Interface and settings live in `ANN.Net.Abstractions`,
  the implementation is `internal` in `ANN.Net/Tokenizers` and is exposed through
  `NetworkFactory.BuildBpeTokenizer`. Classes are `[Serializable]` so a trained
  tokenizer can be persisted the same way the networks are.

## Public surface

```csharp
public interface ITokenizer
{
    int VocabularySize { get; }                              // current size, grows while training
    IReadOnlyDictionary<string, int> SpecialTokens { get; }  // name -> id
    void Train(IEnumerable<string> corpus);                  // learns merges from the texts
    int[] Encode(string text);
    string Decode(IEnumerable<int> tokens);
}

public class BpeTokenizerSettings
{
    public ushort VocabularySize { get; set; } = 1024;  // target, must be >= 256 + special tokens
    public ushort MinimumPairFrequency { get; set; } = 2;
    public IList<string> SpecialTokens { get; set; } = new List<string>();
}

ITokenizer NetworkFactory.BuildBpeTokenizer(BpeTokenizerSettings settings)
```

Token ids are `int`; the one-hot bridge to the transformer's `Quad[][]` input is
shown in the README rather than baked into the tokenizer.

## Implementation

`BpeTokenizer`

- `vocabulary : List<byte[]>` id -> bytes; `merges : Dictionary<(int, int), int>`
  pair -> merged id, in rank order (the id itself is the rank since ids are
  assigned incrementally).
- `Train`: split every text into chunks, count distinct chunks, represent each as
  `List<int>` of byte ids. Loop: count weighted pair frequencies, pick the best
  pair, append a merge, replace the pair inside every chunk. Repeat until the
  vocabulary reaches the target or the best frequency is below the minimum.
- `Encode`: split into chunks; for each chunk apply merges greedily by lowest
  rank until none applies (standard BPE); results are cached per chunk string.
- `Decode`: concatenate the vocabulary bytes of every id and decode as UTF-8.
  Unknown ids throw `ArgumentOutOfRangeException`.

`TextChunker` (static) owns the chunk rule so a future tokenizer can reuse it.

## Error handling

Settings are validated in the constructor (`VocabularySize >= 256 + specials`,
no duplicate or empty special token names). `Encode`/`Decode` reject `null`.

## Testing

xunit tests in `ANN.Net.Tests/Tokenizers`:

- untrained tokenizer encodes to raw UTF-8 bytes and round-trips,
- trained tokenizer round-trips ASCII, Cyrillic and emoji text,
- the most frequent pair is merged first and a frequent word becomes one token,
- merges never cross a chunk boundary,
- training stops early when no pair repeats, and reaches the target otherwise,
- special tokens get the expected ids, decode to their name and are never emitted
  from plain text,
- training is deterministic,
- invalid settings throw.

## Out of scope

Regex-based GPT-2 pre-tokenization, byte fallback tables, vocabulary file formats,
WordPiece/Unigram variants, and a learned embedding layer in front of the
transformer.
