import glob
import json
import logging
import os
import re

from multiprocessing import Pool

import gluonnlp as nlp
import mxnet as mx
import numpy as np


class PTTMxnetDataset():
    def __init__(self, tsv_filename, tokenizer, base_dir='../ptt_dataset', column_indices=None):
        self.dataset = nlp.data.TSVDataset(os.path.join(base_dir, tsv_filename), num_discard_samples=1)
        self.transform = nlp.data.BERTSentenceTransform(
            tokenizer=tokenizer,
            max_seq_length=256,
            pair=False)
        self.column_indices = column_indices
    
    def __iter__(self):
        return iter(self.dataset)

    def _check_sample(self, sample, indices, content_min_len, push_min_len):
        if len(sample[indices[0]]) < content_min_len:
            return False
        
        for idx in indices[1:]:
            if len(sample[idx]) < push_min_len:
                return False
        return True

    @staticmethod
    def filter(text):
        # filter category label
        text = re.sub('\[.{2}\]', '', text)

        # filter "Re:"
        text = re.sub('Re:', '', text)

        # filter spaces
        text = re.sub(' +', '', text)


        return text

    def get(self, content_min_len=20, push_min_len=4):
        non_transformed_dataset = []
        transformed_dataset = []
        for sample in self.dataset:
            transformed_sample = []
            non_transformed_sample = []
            if not self._check_sample(sample, self.column_indices, content_min_len, push_min_len):
                continue
            for idx in self.column_indices:
                if sample[idx]: # prevent blank string.
                    non_transformed_sample += [
                        PTTMxnetDataset.filter(sample[idx])]
                    transformed_sample += self.transform(
                        (PTTMxnetDataset.filter(sample[idx]),))

            non_transformed_dataset.append(
                tuple(non_transformed_sample)
            )
            transformed_dataset.append(
                tuple(transformed_sample)
            )
        return non_transformed_dataset, transformed_dataset
