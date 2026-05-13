#!/usr/bin/env python3
"""
Prepare Kaldi-style manifest files for a LibriTTS-R split stored in HDF5.

Reads utt_ids from the HDF5 file, derives the speaker from the utt_id
(LibriTTS-R naming: {speaker}_{chapter}_{utterance}), and looks up transcripts
from either:

  --transcript_file  A single TSV file with one line per utterance:
                       utt_id<TAB>transcript text
                     (preferred — avoids one file per utterance)

  --libritts_dir     Root of the original LibriTTS-R split directory tree
                     ({speaker}/{chapter}/{utt_id}.normalized.txt).
                     Used as a fallback when --transcript_file is not given.

To create a transcript TSV from an inflated LibriTTS-R split directory:
  find $split_dir -name '*.normalized.txt' | sort | \
    awk -F/ '{utt=substr($NF,1,length($NF)-15); getline t < $0; print utt"\t"t}' \
    > train-clean-100.tsv

Writes to --des_dir:
  wav.scp     "utt_id utt_id"  (path field unused; HDF5 key = utt_id)
  text        "utt_id transcript"
  utt2spk     "utt_id speaker_id"
  spk2utt     "speaker_id utt1 utt2 ..."
"""
import argparse
import logging
import os
import h5py
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_transcripts_from_tsv(tsv_path):
    """Load {utt_id: transcript} from a tab-separated file."""
    utt2text = {}
    with open(tsv_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip('\n')
            if '\t' not in line:
                logger.warning('Line %d in %s has no tab separator; skipping.',
                               lineno, tsv_path)
                continue
            utt_id, text = line.split('\t', 1)
            utt2text[utt_id.strip()] = text
    return utt2text


def load_transcripts_from_dir(libritts_dir, utts):
    """Load {utt_id: transcript} by opening one .normalized.txt per utterance."""
    utt2text = {}
    missing = 0
    for utt in tqdm(utts):
        parts = utt.split('_')
        spk = parts[0] if len(parts) >= 2 else utt
        chapter = parts[1] if len(parts) >= 2 else utt
        txt_path = os.path.join(libritts_dir, spk, chapter,
                                '{}.normalized.txt'.format(utt))
        if not os.path.exists(txt_path):
            logger.warning('Missing transcript: %s', txt_path)
            missing += 1
            continue
        with open(txt_path) as f:
            utt2text[utt] = f.readline().strip()
    if missing:
        logger.warning('%d utterances skipped due to missing transcripts.', missing)
    return utt2text


def main(args):
    if args.transcript_file:
        # Derive utt_ids directly from the TSV — no need to open the HDF5.
        utt2text = load_transcripts_from_tsv(args.transcript_file)
        utts = list(utt2text.keys())
    elif args.libritts_dir:
        if not args.hdf5_file:
            raise ValueError('--hdf5_file is required when using --libritts_dir.')
        with h5py.File(args.hdf5_file, 'r') as h5:
            utts = list(h5.keys())
        utt2text = load_transcripts_from_dir(args.libritts_dir, utts)
    else:
        raise ValueError('Provide either --transcript_file or --libritts_dir.')

    utt2spk, spk2utt = {}, {}
    for utt in utts:
        if utt not in utt2text:
            continue
        parts = utt.split('_')
        spk = parts[0] if len(parts) >= 2 else utt
        utt2spk[utt] = spk
        spk2utt.setdefault(spk, []).append(utt)

    kept = [u for u in utts if u in utt2text]
    logger.info('%d / %d utterances have transcripts.', len(kept), len(utts))

    os.makedirs(args.des_dir, exist_ok=True)
    with open(os.path.join(args.des_dir, 'wav.scp'), 'w') as f:
        for utt in kept:
            f.write('{} {}\n'.format(utt, utt))
    with open(os.path.join(args.des_dir, 'text'), 'w') as f:
        for utt in kept:
            f.write('{} {}\n'.format(utt, utt2text[utt]))
    with open(os.path.join(args.des_dir, 'utt2spk'), 'w') as f:
        for utt in kept:
            f.write('{} {}\n'.format(utt, utt2spk[utt]))
    with open(os.path.join(args.des_dir, 'spk2utt'), 'w') as f:
        for spk, us in spk2utt.items():
            f.write('{} {}\n'.format(spk, ' '.join(us)))

    logger.info('Wrote %d utterances to %s', len(kept), args.des_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_file', type=str, default=None,
                        help='HDF5 file for this split (e.g. train-clean-100.h5). '
                             'Required when using --libritts_dir; not needed with '
                             '--transcript_file.')
    parser.add_argument('--transcript_file', type=str, default=None,
                        help='TSV file with one "utt_id<TAB>transcript" line per '
                             'utterance. Preferred over --libritts_dir.')
    parser.add_argument('--libritts_dir', type=str, default=None,
                        help='Root of the original LibriTTS-R split directory '
                             '(fallback: opens one .normalized.txt per utterance)')
    parser.add_argument('--des_dir', type=str, required=True,
                        help='Output directory for manifest files')
    args = parser.parse_args()
    main(args)
