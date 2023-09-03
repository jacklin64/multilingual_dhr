import argparse
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import faiss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_index_path", type=str)
    parser.add_argument("--transformed_index_dir", type=str)
    args = parser.parse_args()

    ## read oringal index
    with open(args.original_index_path, 'rb') as f:
        corpus_emb, corpus_arg_idx, docids=pickle.load(f)

    emb_dim = corpus_emb.shape[-1]
    index = faiss.IndexFlatIP(emb_dim)

    with open(os.path.join(args.transformed_index_dir, 'docid'), 'w') as id_file:
        for docid in docids:
            id_file.write(f'{docid}\n')

    index.add(new_corpus_emb.astype('float32'))
    faiss.write_index(index, os.path.join(args.transformed_index_dir, 'index'))

	

if __name__ == "__main__":
	main()
