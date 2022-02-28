import argparse
import glob
import sys, os, time, pickle
import time
import torch
import hashlib
import json
import numpy as np

import src.metrics
from src import utils


def merge_results(loaded_sentences, loaded_samples):
    merged_sentences, merged_samples = {}, {}


    merged_sentences['sentences'] = [sentence for file in loaded_sentences for sentence in file['sentences']]
    merged_sentences['is_completed'] = [sentence for file in loaded_sentences for sentence in file['is_completed']]

    merged_samples['samples'] = [sentence for file in loaded_samples for sentence in file['samples']]
    merged_samples['is_completed'] = [sentence for file in loaded_samples for sentence in file['is_completed']]

    n_lst = [1, 2, 3, 4, 5, 6]
    merged_samples['unique_ngram_frac'] = src.metrics.get_unique_ngram_fraction(merged_samples['samples'], n_lst)

    ppls = np.array([file['ppl'] for file in loaded_samples])
    merged_samples['ppl'] = np.exp(np.log(ppls).mean()).item()

    args = loaded_samples[0]['args']
    del args.start_from_generations
    merged_samples['args'] = args

    return merged_sentences, merged_samples


def merge_and_save_results(loaded_sentences, loaded_samples, args):
    merged_sentences, merged_samples = merge_results(loaded_sentences, loaded_samples)

    name = f'{args.datasplit}_p{args.top_p}_k{args.top_k}_t{args.temp}_seed{args.seed}'
    sentences_fname = f'{folder_name}/sentences_{name}.p'
    with open(sentences_fname, 'wb') as f:
        pickle.dump([merged_sentences['sentences'], merged_sentences['is_completed']], f)

    sentences_fname = f'{folder_name}/sample_{name}.p'
    with open(sentences_fname, 'wb') as f:
        pickle.dump([merged_samples['samples'], merged_samples['is_completed'], merged_samples['unique_ngram_frac'], merged_samples['ppl'], merged_samples['args']], f)


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
    assert len(fnames) > 0, 'There are no samples!!'

    name_hash = hashlib.sha1(str.encode(name))
    start_idx = int(name_hash.hexdigest(), 16) % 337735

    # Merge sentences
    results_json = []
    loaded_samples = []
    loaded_sentences = []
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
        results_json += sample_info

        loaded_sentences += [
            {'sentences': sentences, 'is_completed': is_completed}
        ]
        loaded_samples += [
            {'samples': samples, 'is_completed': is_completed,
            'unique_ngram_frac': unique_ngram_frac, 'ppl': ppl, 'args': args}
        ]

        start_idx = start_idx + len(sentences)

    merge_and_save_results(loaded_sentences, loaded_samples, args)

    MODEL_NAMES = {
        'gpt2-xl': 'xl-1542M',
        'gpt2-large': 'large-762M',
        'gpt2-medium': 'medium-345M',
        'gpt2': 'small-117M',
    }

    name = f'{MODEL_NAMES[args.model_name]}-p{args.top_p}'
    fname = f'outputs/seed_{args.seed}/{name}.{args.datasplit}.jsonl'
    with open(fname, 'w') as fp:
        fp.write('\n'.join(json.dumps(x) for x in results_json))

    print('Merged %d samples for partition %s in model %s p%.2f' %
          (len(results_json), args.datasplit, args.model_name, args.top_p))
    print()
