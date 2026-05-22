import argparse
import os
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2

INFERENCE_MODES = ['standard', 'spkemb_only', 'no_spkemb']

INPUTS_DIR = "/gscratch/tial/kpever/workspace/CosyVoice/mari_example_outputs/inputs"

UTT_IDS = [
    "george_party_1",
    "george_party_2",
    "george_party_3",
    "green_ball_1",
    "green_ball_2",
    "janet_broccoli_1",
    "janet_broccoli_2",
    "mary_languages_1",
    "mary_languages_2",
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='path to CosyVoice2 model directory')
    parser.add_argument('--mode', default='standard', choices=INFERENCE_MODES,
                        help='standard: full zero-shot; spkemb_only: x-vector only, no prompt prefix; '
                             'no_spkemb: prompt tokens+mel, zeroed x-vector')
    parser.add_argument('--output_dir', default='mari_example_outputs/standard_outputs',
                        help='directory to write output wavs')
    parser.add_argument('--prosody_encoder_path', default='',
                        help='path to prosodyenc_weights.pt; required when model was trained with continuous prosody encoder')
    return parser.parse_args()


def read_txt(path):
    with open(path) as f:
        return f.read().strip()


def base_id(utt_id):
    """Return the utterance group name, e.g. 'george_party' from 'george_party_1'."""
    return '_'.join(utt_id.rsplit('_', 1)[:-1])


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cosyvoice = CosyVoice2(args.model_path, load_jit=False, load_trt=False,
                           prosody_encoder_path=args.prosody_encoder_path)

    def pick_prompt(utt_id):
        my_base = base_id(utt_id)
        for candidate in UTT_IDS:
            if base_id(candidate) != my_base:
                return candidate
        raise ValueError(f'No valid prompt found for {utt_id}')

    for utt_id in UTT_IDS:
        tts_sentence = read_txt(f"{INPUTS_DIR}/{utt_id}.txt")
        prompt_id = pick_prompt(utt_id)
        prompt_sentence = read_txt(f"{INPUTS_DIR}/{prompt_id}.txt")
        prompt_speech_path = f"{INPUTS_DIR}/{prompt_id}.wav"

        print(f"[{utt_id}] TTS: {tts_sentence}")
        print(f"[{utt_id}] Prompt ({prompt_id}): {prompt_sentence}")

        if args.mode == 'standard':
            out_path = os.path.join(args.output_dir, f"{utt_id}.wav")
            gen = cosyvoice.inference_zero_shot(tts_sentence, prompt_sentence, prompt_speech_path)
        elif args.mode == 'spkemb_only':
            out_path = os.path.join(args.output_dir, f"{args.mode}/{utt_id}.wav")
            gen = cosyvoice.inference_zero_shot_spkemb_only(tts_sentence, prompt_speech_path)
        elif args.mode == 'no_spkemb':
            out_path = os.path.join(args.output_dir, f"{args.mode}/{utt_id}.wav")
            gen = cosyvoice.inference_zero_shot_no_spkemb(tts_sentence, prompt_sentence, prompt_speech_path)

        # mkdir
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        for out in gen:
            torchaudio.save(out_path, out['tts_speech'].cpu(), cosyvoice.sample_rate)


if __name__ == '__main__':
    main()
