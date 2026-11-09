# Actual native T5 tokenizer validation

On 2026-11-09 the exact locally staged FLUX.1-schnell revision
`741f7c3ce8b383c54771c7003378a50191e9efe9` T5TokenizerFast loader completed
with SentencePiece 0.2.3 and protobuf 7.36.2. It encoded all four declared prompt
strings to fixed `[4,256]` coordinates and produced nonempty decoded text.
The actual class was
`transformers.models.t5.tokenization_t5_fast.T5TokenizerFast`.
[The portable evidence record](tokenizer-evidence.json) retains the actual
receipt hashes, two wheel identities and relevant 63-version ledger.

The original pretrained pipeline attempt failed before any teacher row or
update: configured `add_prefix_space` forces slow-to-fast conversion, which
requires SentencePiece; missing protobuf then masked that initial exception.
Both dependencies were absent from the CPU17 environment. That genuine failure
and the original17 random-tiny native CPU checks remain retained.

The repair used a fresh Python3.12 environment with finite explicit read-only
links to the 61 qualified distributions, preserving every version and VCS pin.
It added only the two separately resolved and reviewed public binary wheels.
The old environment and installed product were unchanged. The actual loader
used `local_files_only=True` against the staged `tokenizer_2` directory; it
loaded no neural model weights, initialized no CUDA, made no network calls, and
replayed neither CPU17 nor the old dependency resolution. Its owned CPU8 job
closed with genuine outer0/worker0 and released only its own allocation.

For a fresh compatible environment, install the separate native distribution:
`python -m pip install './native-candidate[test]'`. Its pyproject now declares
both exact tested tokenizer versions alongside the existing numerical/source
pins. With the already staged exact model directory, the focused local check
is the real loader, not a surrogate tokenizer:

```python
from pathlib import Path
from transformers import T5TokenizerFast

model_dir = Path("<your-staged-model-directory>")
tokenizer = T5TokenizerFast.from_pretrained(
    model_dir / "tokenizer_2", local_files_only=True
)
encoded = tokenizer(
    ["A red ceramic teapot on a wooden table."],
    padding="max_length", max_length=256, truncation=True,
)
assert len(encoded["input_ids"][0]) == 256
```

This is tokenizer qualification only. It does not establish pretrained DiT/T5
neural execution, completed training, native output quality or an LR winner.
The validated 01ac numerical code and original top-level preview package stay
unchanged; no version/tag or weight release is introduced.
