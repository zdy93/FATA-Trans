# edited by Dongyu Zhang
from misc.utils import ddict

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertForPreTraining,
    GPT2Config,
    GPT2LMHeadModel
)

from models.tabformer_tokenizer import TabFormerTokenizer
from models.hierarchical import TabFormerEmbeddings, TabStaticFormerEmbeddings
from models.tabformer_bert import TabFormerBertForMaskedLM, TabFormerBertConfig, TabFormerBertForSequenceClassification, TabFormerBertForPretraining
from models.tabformer_gpt2 import TabFormerGPT2LMHeadModel
from models.tabformer_static_bert import TabStaticTimePosFormerBertForMaskedLM, TabStaticTimePosFormerBertForPretraining, TabStaticTimePosFormerBertForClassification, TabStaticFormerBertConfig
from models.classifier import TabRawForClassification, TabEmbeddingsForClassification, TabEmbeddingsLSTMForClassification, TabEmbeddingsStaticLSTMForClassification


class TabFormerBaseModel(PreTrainedModel):
    def __init__(self, hf_model, tab_embeddings, config):
        super().__init__(config)

        self.model = hf_model
        self.tab_embeddings = tab_embeddings

    def forward(self, input_ids, **input_args):
        inputs_embeds = self.tab_embeddings(input_ids)
        return self.model(inputs_embeds=inputs_embeds, **input_args)


class TabFormerPreTrainedModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class TabFormerHierarchicalLM(TabFormerPreTrainedModel):
    base_model_prefix = "bert"

    def __init__(self, config, vocab):
        super().__init__(config)

        self.config = config

        self.tab_embeddings = TabFormerEmbeddings(self.config)
        self.tb_model = TabFormerBertForMaskedLM(self.config, vocab)

    def forward(self, input_ids, **input_args):
        inputs_embeds = self.tab_embeddings(input_ids)
        return self.tb_model(inputs_embeds=inputs_embeds, **input_args)


class TabFormerHierarchicalForPretraining(TabFormerHierarchicalLM):
    base_model_prefix = "bert"

    def __init__(self, config, vocab):
        super().__init__(config, vocab)

        self.config = config
        self.vocab = vocab

        self.tab_embeddings = TabFormerEmbeddings(self.config)
        self.tb_model = TabFormerBertForPretraining(self.config, self.vocab)

    def forward(self, input_ids, **input_args):
        inputs_embeds = self.tab_embeddings(input_ids)
        return {"loss": torch.tensor([-1.0], device=self.tb_model.device),
                "logits": torch.cat((inputs_embeds, self.tb_model(inputs_embeds=inputs_embeds, **input_args)), 1)}
        # return {"row_embeds":inputs_embeds, "seq_embeds":self.tb_model(inputs_embeds=inputs_embeds, **input_args), "loss":torch.tensor([-1.0])}


class TabStaticFormerHierarchicalLM(TabFormerPreTrainedModel):
    base_model_prefix = "bert"

    def __init__(self, config, vocab):
        super().__init__(config)

        self.config = config

        self.static_tab_embeddings = TabStaticFormerEmbeddings(self.config)
        self.tb_model = TabStaticTimePosFormerBertForMaskedLM(
            self.config, vocab)

    def forward(self, input_ids, static_input_ids, **input_args):
        inputs_embeds = self.static_tab_embeddings(input_ids, static_input_ids)
        return self.tb_model(inputs_embeds=inputs_embeds, **input_args)


class TabStaticFormerHierarchicalForPretraining(TabStaticFormerHierarchicalLM):
    base_model_prefix = "bert"

    def __init__(self, config, vocab):
        super().__init__(config, vocab)

        self.config = config
        self.vocab = vocab

        self.static_tab_embeddings = TabStaticFormerEmbeddings(self.config)
        self.tb_model = TabStaticTimePosFormerBertForPretraining(
            self.config, vocab)

    def forward(self, input_ids, static_input_ids, **input_args):
        inputs_embeds = self.static_tab_embeddings(input_ids, static_input_ids)
        return {"loss": torch.tensor([-1.0], device=self.tb_model.device),
                "logits": torch.cat((inputs_embeds, self.tb_model(inputs_embeds=inputs_embeds, **input_args)), 1)}


class TabFormerHierarchicalForClassification(TabFormerPreTrainedModel):
    base_model_prefix = "bert"

    def __init__(self, config, vocab):
        super().__init__(config)

        self.config = config

        self.tab_embeddings = TabFormerEmbeddings(self.config)
        self.tb_model = TabFormerBertForSequenceClassification(
            self.config, vocab)

    def forward(self, input_ids, **input_args):
        inputs_embeds = self.tab_embeddings(input_ids)
        return self.tb_model(inputs_embeds=inputs_embeds, **input_args)


class TabStaticFormerHierarchicalForClassification(TabFormerPreTrainedModel):
    base_model_prefix = "bert"

    def __init__(self, config, vocab):
        super().__init__(config)

        self.config = config

        self.static_tab_embeddings = TabStaticFormerEmbeddings(self.config)
        self.tb_model = TabStaticTimePosFormerBertForClassification(
            self.config, vocab)

    def forward(self, input_ids, static_input_ids, labels, **input_args):
        inputs_embeds = self.static_tab_embeddings(input_ids, static_input_ids)
        return self.tb_model(inputs_embeds=inputs_embeds, labels=labels, **input_args)


class TabEmbedsForClassification:
    def __init__(self, flatten=False, ncols=12, field_hidden_size=None, num_labels=2, seq_len=None,
                 problem_type=None, cls_dropout_prob=0.1, rnn_hs=None, rnn_bd=False, embeds_model_type=None):

        hidden_size = field_hidden_size if flatten else (
            field_hidden_size * ncols)

        self.config = TabFormerBertConfig(ncols=ncols,
                                          hidden_size=hidden_size,
                                          field_hidden_size=field_hidden_size,
                                          flatten=flatten,
                                          num_attention_heads=ncols,
                                          num_labels=num_labels,
                                          seq_len=seq_len,
                                          problem_type=problem_type,
                                          cls_dropout_prob=cls_dropout_prob,
                                          rnn_hs=rnn_hs,
                                          rnn_bd=rnn_bd)

        self.embeds_model_type = embeds_model_type
        self.get_model()

    def get_model(self):
        if self.embeds_model_type == 'lstm':
            self.model = TabEmbeddingsLSTMForClassification(self.config)
        elif self.embeds_model_type == 'mlp':
            self.model = TabEmbeddingsForClassification(self.config)
        elif self.embeds_model_type == 'lstm-split':
            self.model = TabEmbeddingsStaticLSTMForClassification(self.config)


class TabRawDataForClassification:
    def __init__(self, vocab, ncols=12, field_hidden_size=None, num_labels=2, seq_len=None,
                 problem_type=None, cls_dropout_prob=0.1):

        self.vocab = vocab
        self.config = TabFormerBertConfig(vocab_size=len(self.vocab),
                                          ncols=ncols,
                                          field_hidden_size=field_hidden_size,
                                          num_labels=num_labels,
                                          problem_type=problem_type,
                                          cls_dropout_prob=cls_dropout_prob,)

        self.model = TabRawForClassification(self.config)


class TabFormerBertLM:
    def __init__(self, special_tokens, vocab, field_ce=False, flatten=False, ncols=None, field_hidden_size=768,
                 time_pos_type=None):

        self.ncols = ncols
        self.vocab = vocab
        vocab_file = self.vocab.filename
        hidden_size = field_hidden_size if flatten else (
            field_hidden_size * self.ncols)

        self.config = TabFormerBertConfig(vocab_size=len(self.vocab),
                                          ncols=self.ncols,
                                          hidden_size=hidden_size,
                                          field_hidden_size=field_hidden_size,
                                          flatten=flatten,
                                          num_attention_heads=self.ncols,
                                          time_pos_type=time_pos_type)

        self.tokenizer = BertTokenizer(vocab_file,
                                       do_lower_case=False,
                                       **special_tokens)
        self.model = self.get_model(field_ce, flatten)

    def get_model(self, field_ce, flatten):

        if flatten and not field_ce:
            # flattened vanilla BERT
            model = BertForMaskedLM(self.config)
        elif flatten and field_ce:
            # flattened field CE BERT
            model = TabFormerBertForMaskedLM(self.config, self.vocab)
        else:
            # hierarchical field CE BERT
            model = TabFormerHierarchicalLM(self.config, self.vocab)

        return model


class TabStaticFormerBertLM:
    def __init__(self, special_tokens, vocab, field_ce=False, flatten=False, ncols=None, static_ncols=None, field_hidden_size=768,
                 time_pos_type=None):

        self.ncols = ncols
        self.static_ncols = static_ncols
        self.vocab = vocab
        vocab_file = self.vocab.filename
        hidden_size = field_hidden_size if flatten else (
            field_hidden_size * self.ncols)

        self.config = TabStaticFormerBertConfig(vocab_size=len(self.vocab),
                                                ncols=self.ncols,
                                                static_ncols=self.static_ncols,
                                                hidden_size=hidden_size,
                                                field_hidden_size=field_hidden_size,
                                                flatten=flatten,
                                                num_attention_heads=self.ncols,
                                                time_pos_type=time_pos_type)

        self.tokenizer = BertTokenizer(vocab_file,
                                       do_lower_case=False,
                                       **special_tokens)
        self.model = self.get_model(field_ce, flatten)

    def get_model(self, field_ce, flatten):

        if flatten and not field_ce:
            # flattened vanilla BERT
            model = BertForMaskedLM(self.config)
        elif flatten and field_ce:
            # flattened field CE BERT
            model = TabStaticTimePosFormerBertForMaskedLM(
                self.config, self.vocab)
        else:
            # hierarchical field CE BERT
            model = TabStaticFormerHierarchicalLM(self.config, self.vocab)

        return model


class TabStaticFormerBertClassification:
    def __init__(self, special_tokens, vocab, field_ce=False, flatten=False, ncols=None, static_ncols=None, field_hidden_size=768,
                 seq_len=0, num_labels=2, time_pos_type=None, pretrained_dir=None, problem_type=None, fc=False):

        self.ncols = ncols
        self.static_ncols = static_ncols
        self.vocab = vocab
        vocab_file = self.vocab.filename
        self.fc = fc
        hidden_size = field_hidden_size if flatten else (field_hidden_size * self.ncols)
        self.pretrained_dir = pretrained_dir
        self.problem_type = problem_type

        self.config = TabStaticFormerBertConfig(vocab_size=len(self.vocab),
                                                ncols=self.ncols,
                                                static_ncols=self.static_ncols,
                                                hidden_size=hidden_size,
                                                seq_len=seq_len,
                                                field_hidden_size=field_hidden_size,
                                                flatten=flatten,
                                                num_attention_heads=self.ncols,
                                                num_labels=num_labels,
                                                time_pos_type=time_pos_type,
                                                problem_type=problem_type)

        self.tokenizer = BertTokenizer(vocab_file,
                                       do_lower_case=False,
                                       **special_tokens)
        self.model = self.get_model(field_ce, flatten)

    def get_model(self, field_ce, flatten):
        if self.pretrained_dir is not None:
            model = TabStaticFormerHierarchicalForClassification.from_pretrained(
                    self.pretrained_dir, config=self.config, vocab=self.vocab)
        else:
            model = TabStaticFormerHierarchicalForClassification(self.config, self.vocab)
        return model


class TabFormerBertModel:
    def __init__(self, special_tokens, vocab, field_ce=False, flatten=False, ncols=None, field_hidden_size=768, num_labels=2,
                 seq_len=10, pretrained_dir=None, problem_type=None, time_pos_type=None):

        self.ncols = ncols
        self.vocab = vocab
        vocab_file = self.vocab.filename
        self.num_labels = num_labels
        self.seq_len = seq_len
        self.pretrained_dir = pretrained_dir
        self.problem_type = problem_type
        self.time_pos_type = time_pos_type
        hidden_size = field_hidden_size if flatten else (
            field_hidden_size * self.ncols)

        self.config = TabFormerBertConfig(vocab_size=len(self.vocab),
                                          ncols=self.ncols,
                                          hidden_size=hidden_size,
                                          field_hidden_size=field_hidden_size,
                                          flatten=flatten,
                                          num_attention_heads=self.ncols,
                                          num_labels=self.num_labels,
                                          seq_len=self.seq_len,
                                          problem_type=self.problem_type,
                                          time_pos_type=self.time_pos_type)

        self.tokenizer = BertTokenizer(vocab_file,
                                       do_lower_case=False,
                                       **special_tokens)
        self.model = self.get_model(field_ce, flatten)

    def get_model(self, field_ce, flatten):

        if flatten and not field_ce:
            # flattened vanilla BERT
            if self.pretrained_dir is not None:
                model = BertForPreTraining.from_pretrained(
                    self.pretrained_dir, config=self.config)
            else:
                model = BertForPreTraining(self.config)
        elif flatten and field_ce:
            # flattened field CE BERT
            if self.pretrained_dir is not None:
                model = TabFormerBertForPretraining.from_pretrained(
                    self.pretrained_dir, config=self.config, vocab=self.vocab)
            else:
                model = TabFormerBertForPretraining(self.config, self.vocab)
        else:
            # hierarchical field CE BERT
            if self.pretrained_dir is not None:
                model = TabFormerHierarchicalForPretraining.from_pretrained(
                    self.pretrained_dir, config=self.config, vocab=self.vocab)
            else:
                model = TabFormerHierarchicalForPretraining(
                    self.config, self.vocab)

        return model


class TabStaticFormerBert:
    def __init__(self, special_tokens, vocab, field_ce=False, flatten=False, ncols=None, static_ncols=None, field_hidden_size=768, num_labels=2,
                 seq_len=10, pretrained_dir=None, problem_type=None, time_pos_type=None):

        self.ncols = ncols
        self.static_ncols = static_ncols
        self.vocab = vocab
        vocab_file = self.vocab.filename
        self.num_labels = num_labels
        self.seq_len = seq_len
        self.pretrained_dir = pretrained_dir
        self.problem_type = problem_type
        self.time_pos_type = time_pos_type
        hidden_size = field_hidden_size if flatten else (
            field_hidden_size * self.ncols)

        self.config = TabStaticFormerBertConfig(vocab_size=len(self.vocab),
                                                ncols=self.ncols,
                                                static_ncols=self.static_ncols,
                                                hidden_size=hidden_size,
                                                field_hidden_size=field_hidden_size,
                                                flatten=flatten,
                                                num_attention_heads=self.ncols,
                                                num_labels=self.num_labels,
                                                seq_len=self.seq_len,
                                                problem_type=self.problem_type,
                                                time_pos_type=self.time_pos_type)

        self.tokenizer = BertTokenizer(vocab_file,
                                       do_lower_case=False,
                                       **special_tokens)
        self.model = self.get_model(field_ce, flatten)

    def get_model(self, field_ce, flatten):

        if flatten and not field_ce:
            # flattened vanilla BERT
            if self.pretrained_dir is not None:
                model = BertForPreTraining.from_pretrained(
                    self.pretrained_dir, config=self.config)
            else:
                model = BertForPreTraining(self.config)
        elif flatten and field_ce:
            # flattened field CE BERT
            if self.pretrained_dir is not None:
                model = TabStaticTimePosFormerBertForPretraining.from_pretrained(
                    self.pretrained_dir, config=self.config, vocab=self.vocab)
            else:
                model = TabStaticTimePosFormerBertForPretraining(
                    self.config, self.vocab)
        else:
            # hierarchical field CE BERT
            if self.pretrained_dir is not None:
                model = TabStaticFormerHierarchicalForPretraining.from_pretrained(
                    self.pretrained_dir, config=self.config, vocab=self.vocab)
            else:
                model = TabStaticFormerHierarchicalForPretraining(
                    self.config, self.vocab)

        return model


class TabFormerBertForClassification:
    def __init__(self, special_tokens, vocab, field_ce=False, flatten=False, ncols=None, field_hidden_size=768, num_labels=2,
                 seq_len=10, pretrained_dir=None, problem_type=None, time_pos_type=None):

        self.ncols = ncols
        self.vocab = vocab
        vocab_file = self.vocab.filename
        self.num_labels = num_labels
        self.seq_len = seq_len
        self.pretrained_dir = pretrained_dir
        self.problem_type = problem_type
        self.time_pos_type = time_pos_type
        hidden_size = field_hidden_size if flatten else (
            field_hidden_size * self.ncols)

        self.config = TabFormerBertConfig(vocab_size=len(self.vocab),
                                          ncols=self.ncols,
                                          hidden_size=hidden_size,
                                          field_hidden_size=field_hidden_size,
                                          flatten=flatten,
                                          num_attention_heads=self.ncols,
                                          num_labels=self.num_labels,
                                          seq_len=self.seq_len,
                                          problem_type=self.problem_type,
                                          time_pos_type=self.time_pos_type)

        self.tokenizer = BertTokenizer(vocab_file,
                                       do_lower_case=False,
                                       **special_tokens)
        self.model = self.get_model(field_ce, flatten)

    def get_model(self, field_ce, flatten):

        if flatten and not field_ce:
            # flattened vanilla BERT
            if self.pretrained_dir is not None:
                model = BertForSequenceClassification.from_pretrained(
                    self.pretrained_dir, config=self.config)
            else:
                model = BertForSequenceClassification(self.config)
        elif flatten and field_ce:
            # flattened field CE BERT
            if self.pretrained_dir is not None:
                model = TabFormerBertForSequenceClassification.from_pretrained(
                    self.pretrained_dir, config=self.config, vocab=self.vocab)
            else:
                model = TabFormerBertForSequenceClassification(
                    self.config, self.vocab)
        else:
            # hierarchical field CE BERT
            if self.pretrained_dir is not None:
                model = TabFormerHierarchicalForClassification.from_pretrained(
                    self.pretrained_dir, config=self.config, vocab=self.vocab)
            else:
                model = TabFormerHierarchicalForClassification(
                    self.config, self.vocab)

        return model


class TabFormerGPT2:
    def __init__(self, special_tokens, vocab, field_ce=False, flatten=False):

        self.vocab = vocab
        self.config = GPT2Config(vocab_size=len(self.vocab))

        self.tokenizer = TabFormerTokenizer(
            unk_token=special_tokens.unk_token,
            bos_token=special_tokens.bos_token,
            eos_token=special_tokens.eos_token
        )

        self.model = self.get_model(field_ce, flatten)

    def get_model(self, field_ce, flatten):
        if field_ce:
            model = TabFormerGPT2LMHeadModel(self.config, self.vocab)
        else:
            model = GPT2LMHeadModel(self.config)
        if not flatten:
            tab_emb_config = ddict(vocab_size=len(
                self.vocab), hidden_size=self.config.hidden_size)
            model = TabFormerBaseModel(
                model, TabFormerEmbeddings(tab_emb_config))

        return model
