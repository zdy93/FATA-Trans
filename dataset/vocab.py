from collections import OrderedDict
import numpy as np
import copy


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Vocabulary:
    def __init__(self, adap_thres=10000, target_column_name="Is Fraud?"):
        self.unk_token = "[UNK]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.mask_token = "[MASK]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"

        self.adap_thres = adap_thres
        self.adap_sm_cols = set()

        self.target_column_name = target_column_name
        self.special_field_tag = "SPECIAL"

        self.special_tokens = [self.unk_token, self.sep_token, self.pad_token,
                               self.cls_token, self.mask_token, self.bos_token, self.eos_token]

        self.token2id = OrderedDict()  # {field: {token: id}, ...}
        self.id2token = OrderedDict()  # {id : [token,field]}
        self.field_keys = OrderedDict()
        self.static_field_keys = OrderedDict()
        self.dynamic_field_keys = OrderedDict()
        self.token2id[self.special_field_tag] = OrderedDict()

        self.filename = ''  # this field is set in the `save_vocab` method

        for token in self.special_tokens:
            global_id = len(self.id2token)
            local_id = len(self.token2id[self.special_field_tag])

            self.token2id[self.special_field_tag][token] = [global_id, local_id]
            self.id2token[global_id] = [token, self.special_field_tag, local_id]

    def set_id(self, token, field_name, return_local=False, unk_token=-1):
        global_id, local_id = None, None

        if unk_token not in self.token2id[field_name]:
            global_id = len(self.id2token)
            local_id = len(self.token2id[field_name])
            self.token2id[field_name][unk_token] = [global_id, local_id]
            self.id2token[global_id] = [unk_token, field_name, local_id]

        if token not in self.token2id[field_name]:
            global_id = len(self.id2token)
            local_id = len(self.token2id[field_name])

            self.token2id[field_name][token] = [global_id, local_id]
            self.id2token[global_id] = [token, field_name, local_id]
        else:
            global_id, local_id = self.token2id[field_name][token]

        if return_local:
            return local_id

        return global_id

    def get_id(self, token, field_name="", special_token=False, return_local=False):
        global_id, local_id = None, None
        if special_token:
            field_name = self.special_field_tag

        if token in self.token2id[field_name]:
            global_id, local_id = self.token2id[field_name][token]

        else:
            raise Exception(f"token {token} not found in field: {field_name}")

        if return_local:
            return local_id

        return global_id

    def set_field_keys(self, keys):

        for key in keys:
            self.token2id[key] = OrderedDict()
            self.field_keys[key] = None

        self.field_keys[self.special_field_tag] = None  # retain the order of columns

    def set_static_field_keys(self, keys):

        for key in keys:
            self.static_field_keys[key] = None
        self.static_field_keys[self.special_field_tag] = None

    def set_dynamic_field_keys(self, keys):

        for key in keys:
            self.dynamic_field_keys[key] = None
        self.dynamic_field_keys[self.special_field_tag] = None

    def get_field_ids(self, field_name, return_local=False):
        if field_name in self.token2id:
            ids = self.token2id[field_name]
        else:
            raise Exception(f"field name {field_name} is invalid.")

        selected_idx = 0
        if return_local:
            selected_idx = 1
        return [ids[idx][selected_idx] for idx in ids]

    def get_from_global_ids(self, global_ids, what_to_get='local_ids'):
        device = global_ids.device

        def map_global_ids_to_local_ids(gid):
            return self.id2token[gid][2] if gid != -100 else -100

        def map_global_ids_to_tokens(gid):
            return f'{self.id2token[gid][1]}_{self.id2token[gid][0]}' if gid != -100 else '-'

        if what_to_get == 'local_ids':
            return global_ids.cpu().apply_(map_global_ids_to_local_ids).to(device)
        elif what_to_get == 'tokens':
            vectorized_token_map = np.vectorize(map_global_ids_to_tokens)
            new_array_for_tokens = global_ids.detach().clone().cpu().numpy()
            return vectorized_token_map(new_array_for_tokens)
        else:
            raise ValueError("Only 'local_ids' or 'tokens' can be passed as value of the 'what_to_get' parameter.")

    def save_vocab(self, fname):
        self.filename = fname
        with open(fname, "w") as fout:
            for idx in self.id2token:
                token, field, _ = self.id2token[idx]
                token = "%s_%s" % (field, token)
                fout.write("%s\n" % token)

    def get_field_keys(self, remove_target=True, ignore_special=False):
        keys = list(self.field_keys.keys())

        if remove_target and self.target_column_name in keys:
            keys.remove(self.target_column_name)
        if ignore_special:
            keys.remove(self.special_field_tag)
        return keys

    def get_static_field_keys(self, remove_target=True, ignore_special=False):
        keys = list(self.static_field_keys.keys())

        if remove_target and self.target_column_name in keys:
            keys.remove(self.target_column_name)
        if ignore_special:
            keys.remove(self.special_field_tag)
        return keys

    def get_dynamic_field_keys(self, remove_target=True, ignore_special=False):
        keys = list(self.dynamic_field_keys.keys())

        if remove_target and self.target_column_name in keys:
            keys.remove(self.target_column_name)
        if ignore_special:
            keys.remove(self.special_field_tag)
        return keys

    def get_special_tokens(self):
        special_tokens_map = {}
        # TODO : remove the dependency of re-initializing here. retrieve from field_key = SPECIAL
        keys = ["unk_token", "sep_token", "pad_token", "cls_token", "mask_token", "bos_token", "eos_token"]
        for key, token in zip(keys, self.special_tokens):
            token = "%s_%s" % (self.special_field_tag, token)
            special_tokens_map[key] = token

        return AttrDict(special_tokens_map)

    def __len__(self):
        return len(self.id2token)

    def __str__(self):
        str_ = 'vocab: [{} tokens]  [field_keys={}]'.format(len(self), self.field_keys)
        return str_


def merge_vocab(vocab: Vocabulary, other_vocab: Vocabulary) -> Vocabulary:
    final_vocab = copy.deepcopy(vocab)
    new_field_list = [field for field in final_vocab.field_keys.keys() if field != final_vocab.special_field_tag]
    extend_field_list = [field for field in other_vocab.field_keys.keys()
                         if field not in final_vocab.field_keys.keys()]
    new_field_list.extend(extend_field_list)

    final_vocab.field_keys = OrderedDict()
    for field in new_field_list:
        final_vocab.field_keys[field] = None
    final_vocab.field_keys[final_vocab.special_field_tag] = None  # retain the order of columns

    for field in extend_field_list:
        final_vocab.token2id[field] = OrderedDict()
        for token in other_vocab.token2id[field].keys():
            final_vocab.set_id(token, field)

    return final_vocab


def delete_field(vocab: Vocabulary, field_to_delete: str) -> Vocabulary:
    if field_to_delete not in vocab.field_keys.keys():
        raise ValueError(f'{field_to_delete} is not in the given vocab')
    new_vocab = Vocabulary(adap_thres=vocab.adap_thres, target_column_name=vocab.target_column_name)
    new_field_list = [i for i in vocab.field_keys.keys() if i != field_to_delete
                      and i != vocab.special_field_tag]
    new_dynamic_field_list = [i for i in vocab.dynamic_field_keys.keys()
                              if i != field_to_delete and i != vocab.special_field_tag]
    new_static_field_list = [i for i in vocab.static_field_keys.keys()
                             if i != field_to_delete and i != vocab.special_field_tag]
    new_vocab.set_field_keys(new_field_list)
    new_vocab.set_dynamic_field_keys(new_dynamic_field_list)
    new_vocab.set_static_field_keys(new_static_field_list)
    for field in new_field_list:
        for token in vocab.token2id[field].keys():
            new_vocab.set_id(token, field)
    return new_vocab


def add_field_and_ids(vocab: Vocabulary, field_to_add: str, ordered_token_list) -> Vocabulary:
    if field_to_add in vocab.field_keys.keys():
        raise ValueError(f'{field_to_add} is already in the given vocab')
    new_vocab = copy.deepcopy(vocab)
    new_field_list = [field for field in new_vocab.field_keys.keys() if field != new_vocab.special_field_tag]
    new_field_list.append(field_to_add)

    new_vocab.field_keys = OrderedDict()
    for field in new_field_list:
        new_vocab.field_keys[field] = None
    new_vocab.field_keys[new_vocab.special_field_tag] = None  # retain the order of columns

    new_vocab.token2id[field_to_add] = OrderedDict()
    for token in ordered_token_list:
        new_vocab.set_id(token, field_to_add)
    return new_vocab
