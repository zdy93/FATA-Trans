# edited by Dongyu Zhang
import os
from os import path
import pandas as pd
import numpy as np
import math
import tqdm
import pickle
import logging

from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from torch.utils.data.dataset import Dataset
from dataset.vocab import Vocabulary

logger = logging.getLogger(__name__)
log = logger


class IBMPreloadDataset(Dataset):
    """
    This class is for preloading IBM data only
    """

    def __init__(self,
                 user_ids=None,
                 num_bins=10,
                 cached=True,
                 encoder_cached=False,
                 external_encoder_path="",
                 vocab_cached=False,
                 root="./data/credit_card/",
                 fname="card_transaction_train",
                 vocab_dir="checkpoints",
                 fextension="",
                 nrows=None,
                 adap_thres=10 ** 8,
                 get_rids=True,
                 columns_to_select=None):
        """
        :param user_ids: user ids will be used, None means all are needed
        :param num_bins: number of bins will be used to discretize continuous fields
        :param cached: if cached encoded data will be used, if so, encoded data will be loaded
        :param encoder_cached: if cached encoder will be used, if so, encoder will be loaded from external_encoder_path
        :param external_encoder_path: path to the external encoder file
        :param vocab_cached: if cached preloaded vocab files will be used,
        if so, preloaded vocab files under vocab_dir will be loaded
        :param root: directory stores input data
        :param fname: file name of input data
        :param vocab_dir: directory stores external preloaded vocab files (dynamic, time_feature, static)
        or generated preloaded vocab files will be stored under this directory
        :param fextension: file name extension used by preloaded vocab files and encoded data
        :param nrows: number of rows that will be read from encoded data, None means all are needed
        :param adap_thres: threshold for setting adaptive softmax
        :param get_rids: if row ids will be kept in the dataset
        :param columns_to_select: columns that will be kept in the encoded data, None means all are needed
        """
        self.user_ids = user_ids
        self.root = root
        self.fname = fname
        self.nrows = nrows
        self.fextension = f'_{fextension}' if fextension else ''
        self.cached = cached
        self.encoder_cached = encoder_cached
        self.external_encoder_path = external_encoder_path
        self.vocab_cached = vocab_cached
        self.get_rids = get_rids
        self.columns_to_select = columns_to_select

        self.dynamic_vocab = Vocabulary(adap_thres, target_column_name='Is Fraud?')
        self.time_feature_vocab = Vocabulary(adap_thres, target_column_name='')
        self.static_vocab = Vocabulary(adap_thres, target_column_name='')
        self.encoder_fit = {}

        self.trans_table = None
        self.data = []
        self.static_data = []
        self.times = []
        self.pos_ids = []
        self.labels = []
        self.data_sids = []
        self.data_seq_last_rids = []
        self.data_seq_last_labels = []

        self.ncols = None
        self.static_ncols = None
        self.num_bins = num_bins
        self.encoder_path = None

        self.encode_data()
        if self.vocab_cached:
            self.load_vocab(vocab_dir)
        else:
            self.init_vocab()
            self.save_vocab(vocab_dir)

    def init_vocab(self):
        column_names = list(self.trans_table.columns)
        drop_columns = ['rownumber', 'User', 'Card', 'Timestamp']
        dynamic_columns = ['Year', 'Month', 'Day', 'Hour', 'Amount', 'Use Chip', 'Merchant Name', 'Merchant City',
                           'Merchant State', 'Zip', 'MCC', 'Errors?', 'Is Fraud?']
        time_feature_columns = ['timeFeature']
        static_columns = ['avg_dollar_amt', 'std_dollar_amt', 'top_mcc', 'top_chip']
        column_names = [i for i in column_names if i not in drop_columns]
        assert set(column_names) == set(dynamic_columns + time_feature_columns + static_columns), \
            "some non-drop columns are not used"

        self.dynamic_vocab.set_field_keys(dynamic_columns)
        self.time_feature_vocab.set_field_keys(time_feature_columns)
        self.static_vocab.set_field_keys(static_columns)

        for column in dynamic_columns:
            unique_values = self.trans_table[column].value_counts(
                sort=True).to_dict()  # returns sorted
            for val in unique_values:
                self.dynamic_vocab.set_id(val, column)

        for column in time_feature_columns:
            unique_values = self.trans_table[column].value_counts(
                sort=True).to_dict()  # returns sorted
            for val in unique_values:
                self.time_feature_vocab.set_id(val, column)

        for column in static_columns:
            unique_values = self.trans_table[column].value_counts(
                sort=True).to_dict()  # returns sorted
            for val in unique_values:
                self.static_vocab.set_id(val, column)

        log.info(f"total columns: {list(column_names)}, "
                 f"total dynamic columns: {dynamic_columns}, "
                 f"total time feature columns: {time_feature_columns},"
                 f"total static columns: {static_columns}")
        log.info(f"total dynamic vocabulary size: {len(self.dynamic_vocab.id2token)}, "
                 f"total time feature vocabulary size: {len(self.time_feature_vocab.id2token)}, "
                 f"total static vocabulary size: {len(self.static_vocab.id2token)}")

        for vocab in [self.dynamic_vocab, self.time_feature_vocab, self.static_vocab]:
            for column in vocab.field_keys:
                vocab_size = len(vocab.token2id[column])
                log.info(f"column : {column}, vocab size : {vocab_size}")

                if vocab_size > vocab.adap_thres:
                    log.info(f"\tsetting {column} for adaptive softmax")
                    vocab.adap_sm_cols.add(column)

    def save_vocab(self, vocab_dir):
        for vocab, vocab_prefix in zip([self.dynamic_vocab, self.time_feature_vocab, self.static_vocab],
                                       ['dynamic', 'time_feature', 'static']):
            file_name = path.join(vocab_dir, f'{vocab_prefix}_vocab{self.fextension}.nb')
            log.info(f"saving vocab at {file_name}")
            vocab.save_vocab(file_name)
            file_name = path.join(vocab_dir, f'{vocab_prefix}_vocab_ob{self.fextension}')
            with open(file_name, 'wb') as vocab_file:
                pickle.dump(vocab, vocab_file)
            log.info(f"saving vocab object at {file_name}")

    def load_vocab(self, vocab_dir):
        for vocab_prefix in ['dynamic', 'time_feature', 'static']:
            file_name = path.join(vocab_dir, f'{vocab_prefix}_vocab_ob{self.fextension}')
            if not path.exists(file_name):
                raise FileNotFoundError(f"external {vocab_prefix} file is not found")
            else:
                if vocab_prefix == 'dynamic':
                    self.dynamic_vocab = pickle.load(open(file_name, 'rb'))
                elif vocab_prefix == 'time_feature':
                    self.time_feature_vocab = pickle.load(open(file_name, 'rb'))
                else:
                    self.static_vocab = pickle.load(open(file_name, 'rb'))

    @staticmethod
    def label_fit_transform(column, enc_type="label", unk_value=None):
        column = np.asarray(column).reshape(-1, 1)
        if enc_type == "label":
            mfit = OrdinalEncoder(
                handle_unknown='use_encoded_value', unknown_value=unk_value)
        else:
            mfit = RobustScaler()
        mfit.fit(column)

        return mfit, mfit.transform(column)

    @staticmethod
    def fraudEncoder(X):
        fraud = (X == 'Yes').astype(int)
        return pd.DataFrame(fraud)

    @staticmethod
    def nanNone(X):
        return X.where(pd.notnull(X), 'None')

    @staticmethod
    def nanZero(X):
        return X.where(pd.notnull(X), 0)

    def _quantization_binning(self, data):
        qtls = np.arange(0.0, 1.0 + 1 / self.num_bins, 1 / self.num_bins)
        # (num_bins + 1, num_features)
        bin_edges = np.quantile(data, qtls, axis=0)
        bin_widths = np.diff(bin_edges, axis=0)
        bin_centers = bin_edges[:-1] + bin_widths / 2  # ()
        return bin_edges, bin_centers, bin_widths

    def _quantize(self, inputs, bin_edges):
        quant_inputs = np.zeros(inputs.shape[0])
        for i, x in enumerate(inputs):
            quant_inputs[i] = np.digitize(x, bin_edges)
        quant_inputs = quant_inputs.clip(1, self.num_bins) - 1  # Clip edges
        return quant_inputs

    def prepare_samples(self):
        pass

    def get_csv(self, fname):
        data = pd.read_csv(fname, nrows=self.nrows)
        if self.user_ids:
            log.info(f'Filtering data by user ids list: {self.user_ids}...')
            self.user_ids = map(int, self.user_ids)
            data = data[data['User'].isin(self.user_ids)]

        self.nrows = data.shape[0]
        log.info(f"read data : {data.shape}")
        return data

    def write_csv(self, data, fname):
        log.info(f"writing to file {fname}")
        data.to_csv(fname, index=False)

    def encode_data(self):
        dirname = path.join(self.root, "preprocessed")
        fname = f'{self.fname}{self.fextension}.encoded.csv'
        data_file = path.join(self.root, f"{self.fname}.csv")

        if self.cached and path.isfile(path.join(dirname, fname)):
            log.info(f"cached encoded data is read from {fname}")
            self.trans_table = self.get_csv(path.join(dirname, fname))
            encoder_fname = path.join(
                dirname, f'{self.fname}{self.fextension}.encoder_fit.pkl')
            self.encoder_fit = pickle.load(open(encoder_fname, "rb"))
            return
        elif self.cached and not path.isfile(path.join(dirname, fname)):
            raise FileNotFoundError("cached encoded data is not found")

        if self.encoder_cached and path.isfile(self.external_encoder_path):
            log.info(
                f"cached encoder is read from {self.external_encoder_path}")
            self.encoder_fit = pickle.load(
                open(self.external_encoder_path, "rb"))
        elif self.encoder_cached and not path.isfile(self.external_encoder_path):
            raise FileNotFoundError("cached encoder is not found")

        data = self.get_csv(data_file)
        log.info(f"{data_file} is read.")

        log.info("nan resolution.")
        data['Errors?'] = self.nanNone(data['Errors?'])
        data['Is Fraud?'] = self.fraudEncoder(data['Is Fraud?'])
        data['Zip'] = self.nanZero(data['Zip'])
        data['Merchant State'] = self.nanNone(data['Merchant State'])

        log.info("timestamp fit transform")
        data['Timestamp'] = data['total_minutes']
        data['timeFeature'] = data['total_minutes']

        sub_columns = ['Year', 'Month', 'Day', 'Hour', 'Errors?', 'MCC', 'Zip', 'Merchant State',
                       'Merchant City', 'Merchant Name', 'Use Chip', 'top_mcc', 'top_chip']

        unk_value = -1

        log.info("label-fit-transform.")
        for col_name in tqdm.tqdm(sub_columns):
            col_data = data[col_name]
            encoder_name = col_name
            if self.encoder_fit.get(encoder_name) is not None:
                col_fit = self.encoder_fit.get(encoder_name)
                if isinstance(col_fit, OrdinalEncoder):
                    col_data = np.asarray(col_data).reshape(-1, 1)
                col_data = col_fit.transform(col_data)
            else:
                col_fit, col_data = self.label_fit_transform(
                    col_data, unk_value=unk_value)
                self.encoder_fit[col_name] = col_fit
            data[col_name] = col_data

        log.info("amount quant transform")
        amt_columns = ['timeFeature', 'Amount', 'avg_dollar_amt', 'std_dollar_amt']

        for col_name in tqdm.tqdm(amt_columns):
            coldata = np.array(data[col_name])
            if self.encoder_fit.get(col_name) is not None:
                bin_edges, bin_centers, bin_widths = self.encoder_fit.get(col_name)
                data[col_name] = self._quantize(coldata, bin_edges)
            else:
                bin_edges, bin_centers, bin_widths = self._quantization_binning(
                    coldata)
                data[col_name] = self._quantize(coldata, bin_edges)
                self.encoder_fit[col_name] = [bin_edges, bin_centers, bin_widths]

        if self.columns_to_select:
            columns_to_select = self.columns_to_select
        else:
            # keep time info
            columns_to_select = ['rownumber', 'User', 'Card', 'Timestamp', 'Is Fraud?',
                                 'Year', 'Month', 'Day', 'Hour', 'Amount', 'Use Chip', 'Merchant Name',
                                 'Merchant City', 'Merchant State', 'Zip', 'MCC', 'Errors?',
                                 'timeFeature', 'avg_dollar_amt', 'std_dollar_amt', 'top_mcc', 'top_chip']

        if not self.get_rids:
            columns_to_select = columns_to_select[1:]

        self.trans_table = data[columns_to_select]

        log.info(f"writing cached csv to {path.join(dirname, fname)}")
        if not path.exists(dirname):
            os.mkdir(dirname)
        self.write_csv(self.trans_table, path.join(dirname, fname))

        self.encoder_path = path.join(
            dirname, f'{self.fname}{self.fextension}.encoder_fit.pkl')
        log.info(f"writing cached encoder fit to {self.encoder_path}")
        pickle.dump(self.encoder_fit, open(self.encoder_path, "wb"))
