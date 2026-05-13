#!/usr/bin/env python3
"""
Prepare Kaldi-style manifest files for GigaSpeech stored in HDF5.

Expected input files (set via --data_dir):
  segments.h5            waveforms, accessed as h5_file[utt_id][:]
  training.txt           one utt_id per line  (training split)
  validation.txt         one utt_id per line  (validation split)
  utt2spkr.json          {utt_id: speaker_id}
  train.json             {utt_id + ".wav": {"text": ..., "duration": ..., "phoneme": ...}, ...}
  dev.json               same structure for validation split

Writes to --des_dir:
  train/wav.scp     "utt_id utt_id"  (path field unused; HDF5 key = utt_id)
  train/text        "utt_id transcript"
  train/utt2spk     "utt_id speaker_id"
  train/spk2utt     "speaker_id utt1 utt2 ..."
  dev/  (same files for the validation split)
"""
import argparse
import json
import logging
import os
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_transcripts(json_path):
    """Return a {utt_id: transcript} dict.

    JSON keys are '{utt_id}.wav'; we strip the '.wav' suffix so the returned
    dict is keyed by bare utt_id (matching training.txt / validation.txt).
    """
    with open(json_path) as f:
        raw = json.load(f)
    return {k.removesuffix('.wav'): v['text'] for k, v in raw.items()}


def write_split(utt_ids, utt2text, utt2spk, des_dir, dummy_speaker=False):
    os.makedirs(des_dir, exist_ok=True)

    no_transcript = [u for u in utt_ids if u not in utt2text]
    if no_transcript:
        logger.warning('%d utt_ids have no transcript and will be skipped.',
                       len(no_transcript))

    if dummy_speaker:
        # Fall back to utt_id as speaker for utterances missing from utt2spkr.json.
        # Each such utterance becomes its own "speaker", so spk2embedding collapses
        # to utt2embedding — the best conditioning available without real labels.
        missing_spk = [u for u in utt_ids if u not in utt2spk]
        if missing_spk:
            logger.info('%d utt_ids have no speaker label; using utt_id as speaker.',
                        len(missing_spk))
        resolved_spk = {u: utt2spk.get(u, u) for u in utt_ids}
    else:
        no_speaker = [u for u in utt_ids if u not in utt2spk]
        if no_speaker:
            logger.warning('%d utt_ids have no speaker entry and will be skipped.',
                           len(no_speaker))
        resolved_spk = utt2spk

    kept = [u for u in utt_ids if u in utt2text and u in resolved_spk]

    spk2utt = {}
    for utt in kept:
        spk2utt.setdefault(resolved_spk[utt], []).append(utt)

    with open(os.path.join(des_dir, 'wav.scp'), 'w') as f:
        for utt in kept:
            f.write('{} {}\n'.format(utt, utt))
    with open(os.path.join(des_dir, 'text'), 'w') as f:
        for utt in kept:
            f.write('{} {}\n'.format(utt, utt2text[utt]))
    with open(os.path.join(des_dir, 'utt2spk'), 'w') as f:
        for utt in kept:
            f.write('{} {}\n'.format(utt, resolved_spk[utt]))
    with open(os.path.join(des_dir, 'spk2utt'), 'w') as f:
        for spk, utts in spk2utt.items():
            f.write('{} {}\n'.format(spk, ' '.join(utts)))

    logger.info('Wrote %d utterances to %s', len(kept), des_dir)


def read_lines(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def save_prosody_tokens(utt_ids, km_path, des_dir):
    """Read a .km file (one line per utt, space-separated int tokens) and save utt2prosody_token.pt."""
    utt2prosody_token = {}
    with open(km_path) as f:
        for utt_id, line in zip(utt_ids, f):
            tokens = [int(t) for t in line.strip().split()]
            utt2prosody_token[utt_id] = tokens
    torch.save(utt2prosody_token, os.path.join(des_dir, 'utt2prosody_token.pt'))
    logger.info('Saved %d prosody-token entries to %s', len(utt2prosody_token), des_dir)


def main(args):
    utt2spk = json.load(open(os.path.join(args.data_dir, 'utt2spkr.json')))

    train_utts = read_lines(os.path.join(args.data_dir, 'training.txt'))
    dev_utts = read_lines(os.path.join(args.data_dir, 'validation.txt'))

    train_transcripts = load_transcripts(os.path.join(args.data_dir, 'train.json'))
    dev_transcripts = load_transcripts(os.path.join(args.data_dir, 'dev.json'))

    write_split(train_utts, train_transcripts, utt2spk,
                os.path.join(args.des_dir, 'train'))
    write_split(dev_utts, dev_transcripts, utt2spk,
                os.path.join(args.des_dir, 'dev'), dummy_speaker=True)

    train_km = os.path.join(args.data_dir, 'prosody_labels_meanpooled5/training.km')
    if os.path.exists(train_km):
        save_prosody_tokens(train_utts, train_km, os.path.join(args.des_dir, 'train'))
    dev_km = os.path.join(args.data_dir, 'prosody_labels_meanpooled5/validation.km')
    if os.path.exists(dev_km):
        save_prosody_tokens(dev_utts, dev_km, os.path.join(args.des_dir, 'dev'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing segments.h5, training.txt, '
                             'validation.txt, utt2spkr.json, train.json, dev.json')
    parser.add_argument('--des_dir', type=str, required=True,
                        help='Output root; train/ and dev/ subdirs will be created here')
    args = parser.parse_args()
    main(args)
