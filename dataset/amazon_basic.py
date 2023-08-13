# edited by Dongyu Zhang
from os import path
import pandas as pd
import numpy as np
import pickle
import logging

from torch.utils.data.dataset import Dataset
from abc import abstractmethod

logger = logging.getLogger(__name__)
log = logger


class AmazonBasicDataset(Dataset):
    """
    This class is the parent class for all method-specified amazon dataset classes
    """
    def __init__(self,
                 cls_task,
                 reviewer_ids=None,
                 seq_len=10,
                 root="./data/amazon_movie/",
                 fname="Movie_and_TV_5_train",
                 reviewer_level_cached=False,
                 vocab_cached=False,
                 external_vocab_path="",
                 preload_vocab_dir="./data/amazon_movie/",
                 save_vocab_dir="./data/amazon_movie/",
                 preload_fextension="",
                 fextension="",
                 nrows=None,
                 flatten=False,
                 stride=5,
                 return_labels=True,
                 label_category='last_label',
                 pad_seq_first=False,
                 get_rids=True,
                 long_and_sort=True,
                 binary_task=True,
                 resample_method=None,
                 resample_ratio=10,
                 resample_seed=100):

        """
        :param cls_task: if the task needs BERT model
        :param reviewer_ids: reviewer ids will be used, None means all are needed
        :param seq_len: length of model's input sliding window
        :param root: directory stores encoded data
        :param fname: file name of encoded data
        :param reviewer_level_cached: if reviewer level data will be used, if so, cached data will be loaded
        :param vocab_cached: if external final vocab (not the preload vocab) will be used,
        if so, final vocab will be loaded
        :param external_vocab_path: path to the external final vocab file
        :param preload_vocab_dir: directory stores the preloaded vocab files (dynamic, static, time_feature)
        :param save_vocab_dir: directory path where the final vocab will be stored
        :param preload_fextension: file name extension used by encoded data and preloaded vocab files
        :param fextension: file name extension used by cached reviewer level data and final vocab file
        :param nrows: number of rows that will be read from encoded data, None means all are needed
        :param flatten: if the model is a flatten model or not
        :param stride: stride for model's input sliding window
        :param return_labels: if label will be returned when get item for dataset
        :param label_category: category of returned label,
        :param pad_seq_first: if 0 to seq_len-th sliding windows will be pad to build complete sequences,
        if so, first sliding window is the sequence starts from -(seq_len + 1)-th element
        if not, first sliding window is the sequence starts from 0-th element
        :param get_rids: if row ids will be kept in the dataset
        :param long_and_sort: if the encoded data is very long and sorted by reviewer ids and time
        :param binary_task: if the prediction task is a binary classification task,
        if so, return label will be transferred to binary label
        """

        self.cls_task = cls_task
        self.reviewer_ids = reviewer_ids
        self.seq_len = seq_len
        self.root = root
        self.fname = fname
        self.reviewer_level_cached = reviewer_level_cached
        self.vocab_cached = vocab_cached
        self.external_vocab_path = external_vocab_path
        self.preload_vocab_dir = preload_vocab_dir
        self.save_vocab_dir = save_vocab_dir
        self.preload_fextension = f'_{preload_fextension}' if preload_fextension else ''
        self.fextension = f'_{fextension}' if fextension else ''
        self.nrows = nrows
        self.flatten = flatten
        self.trans_stride = stride

        self.return_labels = return_labels
        self.label_category = label_category
        self.pad_seq_first = pad_seq_first
        self.get_rids = get_rids
        self.long_and_sort = long_and_sort
        self.binary_task = binary_task
        self.resample_method = resample_method
        self.resample_ratio = resample_ratio
        self.resample_seed = resample_seed

        self.vocab = None
        self.dynamic_vocab = None
        self.time_feature_vocab = None
        self.static_vocab = None

        self.trans_table = None
        self.data, self.all_data = [], []
        self.static_data, self.all_static_data = [], []
        self.times, self.all_times = [], []
        self.pos_ids, self.all_pos_ids = [], []
        self.labels, self.all_labels = [], []
        self.data_sids, self.all_data_sids = [], []
        self.data_seq_last_rids, self.all_data_seq_last_rids = [], []
        self.data_seq_last_labels, self.all_data_seq_last_labels = [], []

        self.ncols = None
        self.static_ncols = None
        self.vocab_path = None

        self.load_data()
        if self.vocab_cached and path.exists(self.external_vocab_path):
            self.vocab = pickle.load(open(self.external_vocab_path, 'rb'))
            self.prepare_samples()
        elif self.vocab_cached and not path.exists(self.external_vocab_path):
            raise FileNotFoundError(f"external vocab file {self.external_vocab_path} is not found")
        else:
            self.load_vocab()
            self.get_final_vocab()
            self.prepare_samples()
            self.save_vocab()
        if self.resample_method is not None:
            self.resample_data()

    @abstractmethod
    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.data)

    def save_vocab(self):
        file_name = path.join(self.save_vocab_dir, f'vocab{self.fextension}.nb')
        log.info(f"saving vocab at {file_name}")
        self.vocab.save_vocab(file_name)
        if not self.vocab_cached:
            self.vocab_path = path.join(self.save_vocab_dir, f'vocab_ob{self.fextension}')
            with open(self.vocab_path, 'wb') as vocab_file:
                pickle.dump(self.vocab, vocab_file)
            log.info(f"saving vocab object at {self.vocab_path}")

    def load_vocab(self):
        for vocab_prefix in ['dynamic', 'time_feature', 'static']:
            file_name = path.join(self.preload_vocab_dir, f'{vocab_prefix}_vocab_ob{self.preload_fextension}')
            if not path.exists(file_name):
                raise FileNotFoundError(f"external {vocab_prefix} file {file_name} is not found")
            else:
                if vocab_prefix == 'dynamic':
                    self.dynamic_vocab = pickle.load(open(file_name, 'rb'))
                elif vocab_prefix == 'time_feature':
                    self.time_feature_vocab = pickle.load(open(file_name, 'rb'))
                else:
                    self.static_vocab = pickle.load(open(file_name, 'rb'))

    @abstractmethod
    def get_final_vocab(self):
        pass

    def get_csv(self, fname):
        data = pd.read_csv(fname, nrows=self.nrows)
        if self.reviewer_ids:
            log.info(f'Filtering data by reviewer ids list: {self.reviewer_ids}...')
            self.reviewer_ids = map(int, self.reviewer_ids)
            data = data[data['reviewerID'].isin(self.reviewer_ids)]

        self.nrows = data.shape[0]
        log.info(f"read data : {data.shape}")
        return data

    def load_data(self):
        dirname = path.join(self.root, "preprocessed")
        fname = f'{self.fname}{self.preload_fextension}.encoded.csv'

        if path.isfile(path.join(dirname, fname)):
            log.info(f"cached encoded data is read from {fname}")
            self.trans_table = self.get_csv(path.join(dirname, fname))
        else:
            raise FileNotFoundError(f"Encoded data is not found at {path.join(dirname, fname)}")

    @abstractmethod
    def prepare_samples(self):
        pass

    def resample_data(self):
        last_labels = pd.Series(self.data_seq_last_labels)
        if self.binary_task:
            last_labels = (last_labels >= 4).astype('int32')
        pos_idx = last_labels.loc[last_labels == 1].index.values
        neg_idx = last_labels.loc[last_labels == 0].index.values
        if len(pos_idx) > len(neg_idx):
            less_idx = neg_idx
            more_idx = pos_idx
        else:
            less_idx = pos_idx
            more_idx = neg_idx
        rng = np.random.default_rng(self.resample_seed)
        if self.resample_method == "downsample":
            num_more_keep = int(len(less_idx) * self.resample_ratio)
            keep_more_idx = rng.choice(more_idx, num_more_keep, replace=False)
            keep_idx = np.concatenate([less_idx, keep_more_idx])
            keep_idx.sort()
        elif self.resample_method == "upsample":
            num_less_keep = int(len(more_idx) / self.resample_ratio)
            keep_less_idx = rng.choice(less_idx, num_less_keep, replace=True)
            keep_idx = np.concatenate([more_idx, keep_less_idx])
            keep_idx.sort()
        else:
            keep_less_idx = rng.choice(less_idx, int(self.resample_method), replace=False)
            keep_more_idx = rng.choice(more_idx, int(self.resample_method), replace=False)
            keep_idx = np.concatenate([keep_less_idx, keep_more_idx])
            keep_idx.sort()
        if self.data:
            new_data = [self.data[i] for i in range(len(self.data)) if i in keep_idx]
            self.all_data = self.data
            self.data = new_data
        if self.static_data:
            new_static_data = [self.static_data[i] for i in range(len(self.static_data)) if i in keep_idx]
            self.all_static_data = self.static_data
            self.static_data = new_static_data
        if self.times:
            new_times = [self.times[i] for i in range(len(self.times)) if i in keep_idx]
            self.all_times = self.times
            self.times = new_times
        if self.pos_ids:
            new_pos_ids = [self.pos_ids[i] for i in range(len(self.pos_ids)) if i in keep_idx]
            self.all_pos_ids = self.pos_ids
            self.pos_ids = new_pos_ids
        if self.labels:
            new_labels = [self.labels[i] for i in range(len(self.labels)) if i in keep_idx]
            self.all_labels = self.labels
            self.labels = new_labels
        if self.data_sids:
            new_data_sids = [self.data_sids[i] for i in range(len(self.data_sids)) if i in keep_idx]
            self.all_data_sids = self.data_sids
            self.data_sids = new_data_sids
        if self.data_seq_last_rids:
            new_data_seq_last_rids = [self.data_seq_last_rids[i] for i in range(len(self.data_seq_last_rids)) if i in keep_idx]
            self.all_data_seq_last_rids = self.data_seq_last_rids
            self.data_seq_last_rids = new_data_seq_last_rids
        if self.data_seq_last_labels:
            new_data_seq_last_labels = [self.data_seq_last_labels[i] for i in range(len(self.data_seq_last_labels)) if i in keep_idx]
            self.all_data_seq_last_labels = self.data_seq_last_labels
            self.data_seq_last_labels = new_data_seq_last_labels

