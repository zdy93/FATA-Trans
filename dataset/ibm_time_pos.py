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


class IBMWithTimePosDataset(IBMBasicDataset):
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
            return_data_fea = torch.tensor(self.data[index], dtype=torch.long)
        else:
            return_data_fea = torch.tensor(
                self.data[index], dtype=torch.long).reshape(self.seq_len, -1)
        return_time = torch.tensor(self.times[index], dtype=torch.long)
        return_pos_ids = torch.tensor(self.pos_ids[index], dtype=torch.long)
        return_data = (return_data_fea, return_time, return_pos_ids)
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
        self.vocab = merge_vocab(self.dynamic_vocab, self.static_vocab)

    def user_level_split_data(self):
        fname = path.join(
            self.root, f"preprocessed/{self.fname}.user{self.fextension}.pkl")
        trans_data, trans_labels, trans_time = [], [], []
        if self.get_rids:
            trans_rids = []

        if self.user_level_cached and path.isfile(fname):
            log.info(f"loading cached user level data from {fname}")
            cached_data = pickle.load(open(fname, "rb"))
            trans_data = cached_data["trans"]
            trans_labels = cached_data["labels"]
            trans_time = cached_data["time"]
            columns_names = cached_data["columns"]
            bks_exist = False
            if 'RIDs' in cached_data.keys():
                trans_rids = cached_data['RIDs']
                bks_exist = True

        else:
            columns_names = list(self.trans_table.columns)
            other_columns = ['rownumber', 'User', 'Card', 'Timestamp', 'Is Fraud?']
            not_use_columns = ['timeFeature']
            bks_exist = pd.Series(
                [i in columns_names for i in other_columns]).all()
            columns_names = [i for i in columns_names if i not in other_columns and i not in not_use_columns]
            start_idx_list = self.trans_table.index[
                self.trans_table['User'].ne(self.trans_table['User'].shift())]
            end_idx_list = start_idx_list[1:] - 1
            end_idx_list = end_idx_list.append(self.trans_table.index[-1:])
            for ix in tqdm.tqdm(range(len(start_idx_list))):
                start_ix, end_ix = start_idx_list[ix], end_idx_list[ix]
                user_data = self.trans_table.iloc[start_ix:end_ix + 1]
                user_trans, user_label, user_time = [], [], []
                # new assumption: 'rownumber', 'User', 'Card', 'Timestamp', 'Is Fraud?' are the 0-4h columns
                # 'Timestamp' is the 3rd column, we will keep it in user_time
                # 'Is Fraud?' is the 4th column, we will keep it in user_label
                # 'timeFeature' is the -5th column, we will not use it
                if bks_exist:
                    skip_idx = 4
                    if self.get_rids:
                        user_rids = []
                else:
                    skip_idx = 0
                for idx, row in user_data.iterrows():
                    row = list(row)
                    user_trans.extend(row[skip_idx + 1:-5])
                    user_trans.extend(row[-4:])
                    user_label.append(row[skip_idx])
                    user_time.append(row[skip_idx - 1])
                    if self.get_rids and bks_exist:
                        user_rids.append(row[0])
                trans_data.append(user_trans)
                trans_labels.append(user_label)
                trans_time.append(user_time)
                if self.get_rids and bks_exist:
                    trans_rids.append(user_rids)

            with open(fname, 'wb') as cache_file:
                if self.get_rids and bks_exist:
                    pickle.dump({"trans": trans_data, "labels": trans_labels,
                                 "RIDs": trans_rids, "time": trans_time,
                                 "columns": columns_names}, cache_file)
                else:
                    pickle.dump({"trans": trans_data, "labels": trans_labels, "time": trans_time,
                                 "columns": columns_names}, cache_file)

        # convert to str
        if self.get_rids and bks_exist:
            return trans_data, trans_labels, trans_rids, trans_time, columns_names
        else:
            return trans_data, trans_labels, trans_time, columns_names

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

    def prepare_samples(self):
        log.info("preparing user level data...")
        if self.get_rids:
            trans_data, trans_labels, trans_rids, trans_time, columns_names = self.user_level_split_data()
        else:
            trans_data, trans_labels, trans_time, columns_names = self.user_level_split_data()
        log.info("creating transaction samples with vocab")
        for user_idx in tqdm.tqdm(range(len(trans_data))):
            user_row = trans_data[user_idx]
            user_row_ids = self.format_trans(user_row, columns_names)

            user_labels = trans_labels[user_idx]
            user_time = trans_time[user_idx]
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
                        ids = [bos_token] + ids + [eos_token]
                    self.data.append(ids)

                    time_tail = user_time[0:(jdx + self.seq_len)]
                    time_tail = [i - time_tail[0] for i in time_tail]
                    time = [0 for _ in range(jdx, 0)]
                    time.extend(time_tail)
                    self.times.append(time)

                    tail_len = len(time_tail)
                    pos_tail = list(range(0, tail_len))
                    head_len = self.seq_len - tail_len
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
                    ids = [bos_token] + ids + [eos_token]
                self.data.append(ids)

                time_tail = user_time[0:]
                time_tail = [i - time_tail[0] for i in time_tail]
                time = [0 for _ in range(pad_len)]
                time.extend(time_tail)
                self.times.append(time)

                tail_len = len(time_tail)
                pos_tail = list(range(0, tail_len))
                head_len = self.seq_len - tail_len
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
                    ids = [bos_token] + ids + [eos_token]
                self.data.append(ids)
                time = user_time[jdx:(jdx + self.seq_len)]
                time = [_ - time[0] for _ in time]
                self.times.append(time)
                pos_ids = list(range(0, self.seq_len))
                self.pos_ids.append(pos_ids)
                ids = user_labels[jdx:(jdx + self.seq_len)]
                self.labels.append(ids)
                self.data_seq_last_labels.append(ids[-1])

                if self.get_rids:
                    rids = user_rids[jdx:(jdx + self.seq_len)]
                    self.data_sids.append(
                        '_'.join([str(int(_)) for _ in rids]))
                    self.data_seq_last_rids.append(rids[-1])

        assert len(self.data) == len(self.times) == len(self.labels) \
               == len(self.data_sids) == len(self.data_seq_last_labels) == len(self.data_seq_last_rids), \
            f"len data: {len(self.data)}, len times data: {len(self.times)},"\
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
        log.info(f"ncols: {self.ncols}")
        log.info(f"no of samples {len(self.data)}")
