
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

bu_radio_dir = "/gscratch/tial/data/bu_radio"
# pretrained_model_path = "/gscratch/tial/kpever/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B"
# cosyvoice = CosyVoice2(pretrained_model_path, load_jit=True, load_trt=False)

finetuned_model_path = "/gscratch/tial/kpever/workspace/CosyVoice/examples/gigaspeech/cosyvoice2/exp/combined_models/llm_no_prosody_epoch2_flow_no_prosody_epoch29"
cosyvoice = CosyVoice2(finetuned_model_path, load_jit=False, load_trt=False)

utt_ids = ["f1ajrlp1", "f2bjrlp1", "m3bjrlp1"]

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

    for i, j in enumerate(cosyvoice.inference_zero_shot(
        tts_sentence,
        prompt_sentence,
        prompt_speech_path,
    )):
        torchaudio.save(f"bu_radio_example_outputs/standard_outputs/gigaspeech_llm_gigaspeech_flow/{utt_id}.wav", j["tts_speech"].cpu(), cosyvoice.sample_rate)