import json
import numpy as np
import os
import sys
sys.path.append(os.getcwd())

import utils.config as config
from utils.dataset import Dictionary
import argparse
    
def create_dictionary(dataroot):
    dictionary = Dictionary()
    files = [
        'train.json',
        'val.json',
        'test.json',
    ]
    for path in files:
        question_path = os.path.join(dataroot, path)
        print(question_path)
        qs = json.load(open(question_path))
        for q in qs:
            dictionary.tokenize(q['question'], True, True)
            # for answer in q['answers']:
            dictionary.tokenize(q['answer'], True, False)
    return dictionary


def create_glove_embedding_init(idx2word, glove_file):
    """ Using pre-trained glove embedding for questions. """
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('pre-trained embedding dim is {}d'.format(emb_dim))
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = map(float, vals[1:])
        word2emb[word] = np.array(list(vals))
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="slake",
        choices=["slake", "slake-cp", "vg", "vizwiz", "vqacp-v2", "vqace", "vqacp-v1", "vqa-v2", "gqaood"],
        help="choose dataset",
    )
    args = parser.parse_args()
    print(args)
    dataset = args.dataset
    config.dataset = dataset
    config.update_paths(args.dataset)

    d = create_dictionary(config.qa_path)
    d.dump_to_file(config.dict_path)
    d = Dictionary.load_from_file(config.dict_path)
    weights, word2emb = create_glove_embedding_init(d.idx2word, config.glove_path)
    np.save(config.glove_embed_path, weights)
