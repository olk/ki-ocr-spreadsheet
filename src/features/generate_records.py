'''
                    Copyright Oliver Kowalke 2018.
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import csv
import logging
import numpy as np
import os
import shutil

from collections import Counter
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from records_creator import OCRTFRecordsCreator


def analyse_labels(csvs_p):
    max_text_len = 0
    text = ' '
    for csv_p in csvs_p:
        with open(csv_p) as csv_f:
            reader = csv.reader(csv_f, delimiter=';')
            skipped_rows = 0
            for i, row in enumerate(reader):
                # skip header
                if 3 > i or 13 == i:
                    skipped_rows += 1
                    continue
                # skip summary
                if 23 < i:
                    break
                skipped_cols = 0
                for j, label in enumerate(row):
                    # skipp `of` column
                    if 12 == j:
                        skipped_cols += 1
                        continue
                    text += label
                    if len(label) > max_text_len:
                        max_text_len = len(label)
    alphabet = ''.join(sorted(list(set(Counter(text).keys()))))
    return alphabet, max_text_len


def split_data(data_size, jpgs, train_frac, val_frac):
    assert data_size == len(jpgs)
    # create array of indices
    # each index represents one spreadsheet (e.g. jpg)
    indices = np.arange(0, data_size)
    # split indices in training/validation/testing subsets
    train_indices, test_indices, val_indices = np.split(indices, [int(train_frac * len(indices)), int((1 - val_frac) * len(indices))])
    # split jpgs according to the indices
    train_jpgs = jpgs[train_indices[0]:train_indices[-1]+1]
    test_jpgs = jpgs[test_indices[0]:test_indices[-1]+1]
    val_jpgs = jpgs[val_indices[0]:val_indices[-1]+1]
    return train_jpgs, val_jpgs, test_jpgs


def main():
    # environment
    env_path = find_dotenv()
    load_dotenv(dotenv_path=env_path, verbose=True)
    # parameters
    raw_p = Path(os.environ.get('PATH_RAW')).resolve()
    processed_p = Path(os.environ.get('PATH_PROCESSED')).resolve()
    data_size = int(os.environ.get('DATA_SIZE'))
    cohort_size = int(os.environ.get('COHORT_SIZE'))
    train_frac = float(os.environ.get('TRAIN_FRAC'))
    val_frac = float(os.environ.get('VAL_FRAC'))
    # logging
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    # prepare directories
    shutil.rmtree(processed_p, ignore_errors=True)
    processed_p.mkdir()
    # list JPGs
    jpgs_p = sorted(list(raw_p.glob('*.jpg')), key=lambda f: int(f.stem))
    csvs_p = list(raw_p.glob('*.csv'))
    # each JPG has a CSV
    assert len(jpgs_p) == len(csvs_p)
    # extract alphabet, max_text_len
    alphabet, max_text_len = analyse_labels(csvs_p) 
    # split data into training, validation and test
    train_jpgs_p, val_jpgs_p, test_jpgs_p = split_data(data_size, jpgs_p, train_frac, val_frac)
    # create training records
    tfrecords_creator = OCRTFRecordsCreator(cohort_size, alphabet, max_text_len)
    tfrecords_creator.create(processed_p.joinpath('train'), train_jpgs_p)
    tfrecords_creator.create(processed_p.joinpath('val'), val_jpgs_p)
    tfrecords_creator.create(processed_p.joinpath('test'), test_jpgs_p)


if __name__ == '__main__':
    main()
