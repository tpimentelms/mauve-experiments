import argparse
import glob
import sys, os, time, pickle
import time
import torch
import hashlib
import json

from src import utils


if __name__ == '__main__':
    parser = utils.make_basic_parser()
    args = parser.parse_args()
    print(args)

    # Get file and folder names
    save_directory = f'./outputs/{utils.get_dataset_name_from_datapath(args.data_dir)}_{utils.get_model_basename(args.model_name)}'
    name = f'{args.datasplit}_p{args.top_p}_k{args.top_k}_t{args.temp}_seed{args.seed}__start[0-9][0-9][0-9][0-9][0-9][0-9]'
    folder_name = f'{save_directory}/generations/basic'

    # Get list of files with text samples
    fnames_re = f'{folder_name}/sentences_{name}.p'
    fnames = sorted(glob.glob(fnames_re))

    name_hash = hashlib.sha1(str.encode(name))
    start_idx = int(name_hash.hexdigest(), 16) % 337735

    # Merge sentences
    results = []
    for fname in fnames:
        with open(fname, 'rb') as f:
            sentences, is_completed = pickle.load(f)
        with open(fname.replace('/sentences_', '/sample_'), 'rb') as f:
            samples, is_completed, unique_ngram_frac, ppl, args = pickle.load(f)

        # Get sample info from files
        idx = list(range(start_idx, start_idx + len(sentences)))
        n_tokens = [len(x) for x in samples]
        sample_info = {
            'id': idx,
            'ended': is_completed,
            'length': n_tokens,
            'text': sentences,
        }

        # Convert to list of dictionaries
        sample_info = [dict(zip(sample_info, t)) for t in zip(*sample_info.values())]
        results += sample_info

        start_idx = start_idx + len(sentences)

    MODEL_NAMES = {
        'gpt2-xl': 'xl-1542M',
        'gpt2-large': 'large-762M',
        'gpt2-medium': 'medium-345M',
        'gpt2': 'small-117M',
    }

    name = f'{MODEL_NAMES[args.model_name]}-p{args.top_p}'
    fname = f'outputs/{name}.{args.datasplit}.jsonl'
    with open(fname, 'w') as fp:
        fp.write('\n'.join(json.dumps(x) for x in results))