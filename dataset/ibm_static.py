# edited by Dongyu Zhang
from os import path
import pandas as pd
import tqdm
import pickle
import logging
import torch

from dataset.vocab import merge_vocab
from misc.utils import divide_chunks
from dataset.ibm_basic import IBMBasicDataset

logger = logging.getLogger(__name__)
log = logger


class IBMWithStaticSplitDataset(IBMBasicDataset):
    def __init__(self,
                 cls_task,
                 user_ids=None,
                 seq_len=10,
                 root="./data/amazon_movie/",
                 fname="Movie_and_TV_5_train",
                 user_level_cached=False,
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
                 resample_method=None,
                 resample_ratio=2,
                 resample_seed=100
                 ):

        super().__init__(cls_task,
                         user_ids,
                         seq_len,
                         root,
                         fname,
                         user_level_cached,
                         vocab_cached,
                         external_vocab_path,
                         preload_vocab_dir,
                         save_vocab_dir,
                         preload_fextension,
                         fextension,
                         nrows,
                         flatten,
                         stride,
                         return_labels,
                         label_category,
                         pad_seq_first,
                         get_rids,
                         long_and_sort,
                         resample_method,
                         resample_ratio,
                         resample_seed
                         )

    def __getitem__(self, index):
        if self.flatten:
            return_dynamic_data = torch.tensor(self.data[index], dtype=torch.long)
        else:
            return_dynamic_data = torch.tensor(
                self.data[index], dtype=torch.long).reshape(self.seq_len, -1)
        return_static_data = torch.tensor(self.static_data[index], dtype=torch.long)
        return_pos_ids = torch.tensor(self.pos_ids[index], dtype=torch.long)
        return_type_ids = torch.cat((torch.zeros(1, dtype=torch.long), torch.ones(self.seq_len, dtype=torch.long)),
                                    dim=0)
        return_data = (return_dynamic_data, return_static_data, return_pos_ids, return_type_ids)
        if self.return_labels:
            # only consider sequence_label and last_label
            return_data_label = torch.tensor(
                self.labels[index], dtype=torch.long).reshape(self.seq_len, -1)
            if self.label_category == "sequence_label":
                return_data = return_data + (return_data_label,)
            else:
                return_data = return_data + (return_data_label[-1, :],)

        return return_data

    def get_final_vocab(self):
        vocab = merge_vocab(self.dynamic_vocab, self.time_feature_vocab)
        self.vocab = merge_vocab(vocab, self.static_vocab)


    def user_level_split_data(self):
        fname = path.join(
            self.root, f"preprocessed/{self.fname}.user{self.fextension}.pkl")
        trans_data, trans_labels = [], []
        if self.get_rids:
            trans_rids = []

        if self.user_level_cached  and path.isfile(fname):
            log.info(f"loading cached user level data from {fname}")
            cached_data = pickle.load(open(fname, "rb"))
            trans_data = cached_data["trans"]
            trans_labels = cached_data["labels"]
            columns_names = cached_data["columns"]
            static_columns = cached_data["static_columns"]
            self.static_ncols = len(static_columns) + 1
            bks_exist = False
            if 'RIDs' in cached_data.keys():
                trans_rids = cached_data['RIDs']
                bks_exist = True

        else:
            columns_names = list(self.trans_table.columns)
            other_columns = ['rownumber', 'User', 'Card', 'Timestamp', 'Is Fraud?']
            static_columns = ['avg_dollar_amt', 'std_dollar_amt', 'top_mcc', 'top_chip']
            self.static_ncols = len(static_columns) + 1

            bks_exist = pd.Series(
                [i in columns_names for i in other_columns]).all()
            columns_names = [i for i in columns_names if i not in other_columns and i not in static_columns]
            start_idx_list = self.trans_table.index[
                self.trans_table['User'].ne(self.trans_table['User'].shift())]
            end_idx_list = start_idx_list[1:] - 1
            end_idx_list = end_idx_list.append(self.trans_table.index[-1:])
            for ix in tqdm.tqdm(range(len(start_idx_list))):
                start_ix, end_ix = start_idx_list[ix], end_idx_list[ix]
                user_data = self.trans_table.iloc[start_ix:end_ix + 1]
                user_trans_static, user_trans, user_label = [], [], []
                # new assumption: 'rownumber', 'User', 'Card', 'Timestamp', 'Is Fraud?' are the 0-4h columns
                # 'Is Fraud?' is the 4th column, we will keep it in user_label
                # 'avg_dollar_amt', 'std_dollar_amt', 'top_mcc', 'top_chip' are the -4 - -1th columns
                if bks_exist:
                    skip_idx = 4
                    if self.get_rids:
                        user_rids = []
                else:
                    skip_idx = 0
                # get static feature from the first row in the sequence
                start_static = self.trans_table.iloc[start_ix][-4:]
                user_trans_static.extend(start_static)
                for idx, row in user_data.iterrows():
                    row = list(row)
                    user_trans.extend(row[skip_idx + 1:-4])
                    user_label.append(row[skip_idx])
                    if self.get_rids and bks_exist:
                        user_rids.append(row[0])
                trans_data.append((user_trans_static, user_trans))
                trans_labels.append(user_label)
                if self.get_rids and bks_exist:
                    trans_rids.append(user_rids)

            with open(fname, 'wb') as cache_file:
                if self.get_rids and bks_exist:
                    pickle.dump({"trans": trans_data, "labels": trans_labels,
                                 "RIDs": trans_rids, "columns": columns_names,
                                 "static_columns": static_columns}, cache_file)
                else:
                    pickle.dump({"trans": trans_data, "labels": trans_labels,
                                 "columns": columns_names, "static_columns": static_columns}, cache_file)

        self.vocab.set_static_field_keys(static_columns)
        self.vocab.set_dynamic_field_keys(columns_names)
        # convert to str
        if self.get_rids and bks_exist:
            return trans_data, trans_labels, trans_rids, columns_names, static_columns
        else:
            return trans_data, trans_labels, columns_names, static_columns

    def format_trans(self, trans_lst, column_names):
        trans_lst = list(divide_chunks(
            trans_lst, len(column_names)))
        pan_vocab_ids = []

        sep_id = self.vocab.get_id(self.vocab.sep_token, special_token=True)

        for trans in trans_lst:
            vocab_ids = []
            for jdx, field in enumerate(trans):
                vocab_id = self.vocab.get_id(field, column_names[jdx])
                vocab_ids.append(vocab_id)

            # TODO : need to handle ncols when sep is not added
            # and self.flatten:  # only add [SEP] for BERT + flatten scenario
            if self.cls_task:
                vocab_ids.append(sep_id)

            pan_vocab_ids.append(vocab_ids)

        return pan_vocab_ids

    def format_static_trans(self, static_row, static_columns):
        user_static_ids = []
        sep_id = self.vocab.get_id(self.vocab.sep_token, special_token=True)
        for jdx, field in enumerate(static_row):
            static_vocab_id = self.vocab.get_id(field, static_columns[jdx])
            user_static_ids.append(static_vocab_id)

        # TODO : need to handle ncols when sep is not added
        # and self.flatten:  # only add [SEP] for BERT + flatten scenario
        if self.cls_task:
            user_static_ids.append(sep_id)

        return user_static_ids

    def prepare_samples(self):
        log.info("preparing user level data...")
        if self.get_rids:
            trans_data, trans_labels, trans_rids, columns_names, static_columns = self.user_level_split_data()
        else:
            trans_data, trans_labels, columns_names, static_columns = self.user_level_split_data()
        log.info("creating transaction samples with vocab")
        for user_idx in tqdm.tqdm(range(len(trans_data))):
            user_static_row, user_row = trans_data[user_idx]
            user_static_ids = self.format_static_trans(user_static_row, static_columns)
            user_row_ids = self.format_trans(user_row, columns_names)

            user_labels = trans_labels[user_idx]
            if self.get_rids:
                user_rids = trans_rids[user_idx]

            bos_token = self.vocab.get_id(
                self.vocab.bos_token, special_token=True)  # will be used for GPT2
            eos_token = self.vocab.get_id(
                self.vocab.eos_token, special_token=True)  # will be used for GPT2
            # will be used for padding sequence
            pad_token = self.vocab.get_id(
                self.vocab.pad_token, special_token=True)

            # Padding tokens for first few transaction sequences shorter than self.seq_len
            if self.pad_seq_first:
                for jdx in range(0 - self.seq_len + 1, min(0, len(user_row_ids) - self.seq_len + 1), self.trans_stride):
                    ids_tail = user_row_ids[0:(jdx + self.seq_len)]
                    ncols = len(ids_tail[0])
                    # flattening
                    ids_tail = [idx for ids_lst in ids_tail for idx in ids_lst]
                    ids = [pad_token for _ in range(
                        ncols) for _ in range(jdx, 0)]
                    ids.extend(ids_tail)
                    # for GPT2, need to add [BOS] and [EOS] tokens
                    if not self.cls_task and self.flatten:
                        ids = [bos_token] + user_static_ids + ids + [eos_token]
                    self.data.append(ids)

                    self.static_data.append(user_static_ids)

                    tail_len = jdx + self.seq_len
                    pos_tail = list(range(0, tail_len))
                    # add a position for static features
                    head_len = self.seq_len - tail_len + 1
                    pos_head = [0 for _ in range(head_len)]
                    pos_ids = pos_head + pos_tail
                    self.pos_ids.append(pos_ids)

                    ids_tail = user_labels[0:(jdx + self.seq_len)]
                    ids = [0 for _ in range(jdx, 0)]
                    ids.extend(ids_tail)
                    self.labels.append(ids)
                    self.data_seq_last_labels.append(ids[-1])

                    if self.get_rids:
                        rids_tail = user_rids[0:(jdx + self.seq_len)]
                        rids = [-1 for _ in range(jdx, 0)]
                        rids.extend(rids_tail)
                        self.data_sids.append(
                            '_'.join([str(int(_)) for _ in rids]))
                        self.data_seq_last_rids.append(rids[-1])
            elif not self.pad_seq_first and len(user_row_ids) < self.seq_len:
                pad_len = self.seq_len - len(user_row_ids)
                ids_tail = user_row_ids[0:]
                ncols = len(ids_tail[0])
                # flattening
                ids_tail = [idx for ids_lst in ids_tail for idx in ids_lst]
                ids = [pad_token for _ in range(
                    ncols) for _ in range(pad_len)]
                ids.extend(ids_tail)
                # for GPT2, need to add [BOS] and [EOS] tokens
                if not self.cls_task and self.flatten:
                    ids = [bos_token] + user_static_ids + ids + [eos_token]
                self.data.append(ids)

                self.static_data.append(user_static_ids)

                tail_len = len(user_row_ids)
                pos_tail = list(range(tail_len))
                # add a position for static features
                head_len = pad_len + 1
                pos_head = [0 for _ in range(head_len)]
                pos_ids = pos_head + pos_tail
                self.pos_ids.append(pos_ids)

                ids_tail = user_labels[0:]
                ids = [0 for _ in range(pad_len)]
                ids.extend(ids_tail)
                self.labels.append(ids)
                self.data_seq_last_labels.append(ids[-1])

                if self.get_rids:
                    rids_tail = user_rids[0:]
                    rids = [-1 for _ in range(pad_len)]
                    rids.extend(rids_tail)
                    self.data_sids.append(
                        '_'.join([str(int(_)) for _ in rids]))
                    self.data_seq_last_rids.append(rids[-1])

            for jdx in range(0, len(user_row_ids) - self.seq_len + 1, self.trans_stride):
                ids = user_row_ids[jdx:(jdx + self.seq_len)]
                ids = [idx for ids_lst in ids for idx in ids_lst]  # flattening
                # for GPT2, need to add [BOS] and [EOS] tokens
                if not self.cls_task and self.flatten:
                    ids = [bos_token] + user_static_ids + ids + [eos_token]
                self.data.append(ids)
                self.static_data.append(user_static_ids)

                pos_ids = [0] + list(range(0, self.seq_len))
                self.pos_ids.append(pos_ids)
                ids = user_labels[jdx:(jdx + self.seq_len)]
                self.labels.append(ids)
                self.data_seq_last_labels.append(ids[-1])

                if self.get_rids:
                    rids = user_rids[jdx:(jdx + self.seq_len)]
                    self.data_sids.append(
                        '_'.join([str(int(_)) for _ in rids]))
                    self.data_seq_last_rids.append(rids[-1])

        assert len(self.data) == len(self.static_data) == len(self.labels) \
               == len(self.data_sids) == len(self.data_seq_last_labels) == len(self.data_seq_last_rids), \
            f"len data: {len(self.data)}, len static data: {len(self.static_data)},"\
            f"len labels: {len(self.labels)}, len data sids: {len(self.data_sids)}," \
            f"len data seq_last_rids: {len(self.data_seq_last_rids)}" \
            f"len data seq_last_labels: {len(self.data_seq_last_labels)}"

        '''
            ncols = total fields - 1 (special tokens) - 1 (label)
            if bert:
                ncols += 1 (for sep)
        '''
        self.ncols = len(self.vocab.field_keys) - 2 + \
                     (1 if self.cls_task else 0)
        self.ncols = self.ncols - (self.static_ncols - 1)
        log.info(f"ncols: {self.ncols}")
        log.info(f"no of samples {len(self.data)}")
