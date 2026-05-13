
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

bu_radio_dir = "/gscratch/tial/data/bu_radio"
model_path = "/gscratch/tial/kpever/workspace/CosyVoice/examples/gigaspeech/cosyvoice2/exp/combined_models/llm_w_prosody_epoch3_flow_w_prosody_epoch29"

utt_ids = ["f1ajrlp1", "f2bjrlp1", "m3bjrlp1"]

cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False)

def get_bu_radio_sentence(txt_path):
    with open(txt_path, "r") as f:
        full_str = f.read()
    full_str = " ".join(full_str.split())
    sentences = full_str.split(". ")

    return sentences

def get_sentence_from_txt_path(txt_path):
    with open(txt_path, "r") as f:
        full_str = f.read()
    return full_str.strip()

def get_prosody_tokens_from_txt_path(txt_path):
    with open(txt_path, "r") as f:
        full_str = f.readline().strip()
    tokens = [int(x) for x in full_str.split()]

    return tokens

for utt_id in utt_ids:
    
    spkr = utt_id[:3]

    tts_sentence = get_sentence_from_txt_path(f"/gscratch/tial/kpever/workspace/CosyVoice/bu_radio_example_outputs/inputs/{utt_id}_trimmed.txt")
    print(f"TTS sentence: {tts_sentence}")

    prompt_sentence = get_sentence_from_txt_path(f"/gscratch/tial/kpever/workspace/CosyVoice/bu_radio_example_outputs/inputs/{utt_id}_prompt.txt")
    print(f"Prompt sentence: {prompt_sentence}")

    prompt_speech_path = f"/gscratch/tial/kpever/workspace/CosyVoice/bu_radio_example_outputs/inputs/{utt_id}_prompt.wav"

    prompt_tokens = get_prosody_tokens_from_txt_path(f"/gscratch/tial/kpever/workspace/CosyVoice/bu_radio_example_outputs/inputs/{utt_id}_prompt_tokens.txt")

    for prosody_utt_id in utt_ids:
        prosody_tokens = get_prosody_tokens_from_txt_path(f"/gscratch/tial/kpever/workspace/CosyVoice/bu_radio_example_outputs/inputs/{prosody_utt_id}_trimmed_speech_tokens.txt")

        prompt_tts_tokens = prompt_tokens + prosody_tokens

        for i, j in enumerate(cosyvoice.inference_zero_shot_with_prosody_tokens(
            tts_sentence,
            prompt_sentence,
            prompt_speech_path,
            prosody_tokens=prompt_tts_tokens,
        )):
            torchaudio.save(f"bu_radio_example_outputs/prosody_transfer_outputs/gigaspeech_llm_w_prosody_gigaspeech_flow_w_prosody/{utt_id}_prosody_from_{prosody_utt_id}.wav", j["tts_speech"].cpu(), cosyvoice.sample_rate)