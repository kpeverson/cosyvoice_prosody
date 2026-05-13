# CosyVoice2 with Prosody Conditioning

Fork of [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) extending CosyVoice2 with prosody conditioning.

See the [upstream README](https://github.com/FunAudioLLM/CosyVoice) for setup and general usage.

---

## Prosody Conditioning

### Approach 1 — discrete prosody tokens

A set of learned prosody token embeddings are mixed into the input sequence alongside speech tokens for LLM training, and added for cross-attention in the flow-matching training. Tokens are extracted offline and stored in the parquet dataset.

Relevant config keys (`cosyvoice2_prosody_encoder.yaml`):
```yaml
num_prosody_tokens: 100
prosody_mix_ratio: 8
```

### Approach 2 — Continuous prosody encoder (frozen)

A pretrained [ProsodyvecModel](https://github.com/kpeverson/speaker_disentangled_prosody) is loaded as a frozen sub-module to both the LLM and the flow encoder. At training and inference time, raw audio is processed through a glottal source extractor and then encoded into continuous prosody embeddings that condition generation.

**Pipeline**: raw audio → resample to 16 kHz → `GlottalExtractor` (LPC inverse filter + 1 kHz LPF) → `ProsodyEncoder` → 5× mean-pool (62.5 Hz → 12.5 Hz) → linear projection → LLM input / flow cross-attention

The prosody encoder weights are loaded from a plain PyTorch checkpoint (extracted from the original fairseq checkpoint):
```
/gscratch/tial/kpever/workspace/prosodyvec/prosodyenc_weights.pt
```

Implementation lives in `cosyvoice/prosody/`:
- `glottal.py` — `GlottalExtractor`, `extract_glottal_source()`
- `prosody_encoder.py` — `ProsodyEncoder`, `ProsodyEncoder.from_checkpoint()`

---

## Training

Training scripts are in `examples/gigaspeech/cosyvoice2/local/`.

```sh
bash local/run_cosyvoice2_{llm,flow}_training_w_prosody_{tokens,encoder}.sh
```

---

## Inference

```python
from cosyvoice.cli.cosyvoice import CosyVoice2
import torchaudio

model = CosyVoice2('pretrained_models/CosyVoice2-0.5B')

prompt_wav, sr = torchaudio.load('prompt.wav')

for audio in model.inference_zero_shot_with_prosody_encoder(
    tts_text='Hello world.',
    prompt_text='This is the prompt transcript.',
    prompt_wav=prompt_wav,
    prompt_wav_sr=sr,
):
    # audio is a [1, T] float32 tensor at 24 kHz
    ...
```

`inference_zero_shot_with_prosody_encoder()` extracts the glottal source from `prompt_wav` once and passes the resulting prosody embedding to both the LLM and flow model for all generated chunks.

---

## Acknowledgements

Built on [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice). See upstream repo for full acknowledgements and citations.
