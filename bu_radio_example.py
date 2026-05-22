import argparse
import os
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2

INFERENCE_MODES = ['standard', 'spkemb_only', 'no_spkemb']

UTT_IDS = ["f1ajrlp1", "f2bjrlp1", "m3bjrlp1"]
INPUTS_DIR = "/gscratch/tial/kpever/workspace/CosyVoice/bu_radio_example_outputs/inputs"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='path to CosyVoice2 model directory')
    parser.add_argument('--mode', default='standard', choices=INFERENCE_MODES,
                        help='standard: full zero-shot; spkemb_only: x-vector only, no prompt prefix; no_spkemb: prompt tokens+mel, zeroed x-vector')
    parser.add_argument('--output_dir', default='bu_radio_example_outputs/standard_outputs', help='directory to write output wavs')
    parser.add_argument('--prosody_encoder_path', default='',
                        help='path to prosodyenc_weights.pt; required when model was trained with continuous prosody encoder')
    return parser.parse_args()


def read_txt(path):
    with open(path) as f:
        return f.read().strip()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cosyvoice = CosyVoice2(args.model_path, load_jit=False, load_trt=False, prosody_encoder_path=args.prosody_encoder_path)

    for utt_id in UTT_IDS:
        tts_sentence = read_txt(f"{INPUTS_DIR}/{utt_id}_trimmed.txt")
        prompt_sentence = read_txt(f"{INPUTS_DIR}/{utt_id}_prompt.txt")
        prompt_speech_path = f"{INPUTS_DIR}/{utt_id}_prompt.wav"

        print(f"[{utt_id}] TTS: {tts_sentence}")
        print(f"[{utt_id}] Prompt: {prompt_sentence}")

        if args.mode == 'standard':
            gen = cosyvoice.inference_zero_shot(tts_sentence, prompt_sentence, prompt_speech_path)
        elif args.mode == 'spkemb_only':
            gen = cosyvoice.inference_zero_shot_spkemb_only(tts_sentence, prompt_speech_path)
        elif args.mode == 'no_spkemb':
            gen = cosyvoice.inference_zero_shot_no_spkemb(tts_sentence, prompt_sentence, prompt_speech_path)

        for i, out in enumerate(gen):
            out_path = os.path.join(args.output_dir, f"{utt_id}.wav")
            torchaudio.save(out_path, out['tts_speech'].cpu(), cosyvoice.sample_rate)


if __name__ == '__main__':
    main()
