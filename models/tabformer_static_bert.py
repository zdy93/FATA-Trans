# edited by Dongyu Zhang
import torch
from torch import nn
from packaging import version
import math
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

try:
    from transformers.modeling_bert import ACT2FN, BertLayerNorm
    from transformers.modeling_bert import BertEncoder, BertPooler, BertModel, BertPreTrainedModel, BertForMaskedLM, \
        BertForSequenceClassification
    from transformers.configuration_bert import BertConfig
except ModuleNotFoundError:
    from transformers.models.bert.modeling_bert import ACT2FN

    BertLayerNorm = nn.LayerNorm
    from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertModel, BertPreTrainedModel, \
        BertForMaskedLM, BertForSequenceClassification
    from transformers.models.bert.configuration_bert import BertConfig

from models.custom_criterion import CustomAdaptiveLogSoftmax
from typing import List, Optional, Tuple, Union


class TabFormerBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.field_hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class TabFormerBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = TabFormerBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class TabFormerBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = TabFormerBertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class TabStaticFormerBertConfig(BertConfig):
    def __init__(
            self,
            flatten=True,
            ncols=11,
            static_ncols=6,
            vocab_size=30522,
            field_hidden_size=64,
            hidden_size=768,
            num_attention_heads=12,
            pad_token_id=0,
            num_labels=2,
            seq_len=10,
            problem_type=None,
            type_vocab_size=2,
            time_pos_type=None,
            **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.ncols = ncols
        self.static_ncols = static_ncols
        self.field_hidden_size = field_hidden_size
        self.hidden_size = hidden_size
        self.flatten = flatten
        self.vocab_size = vocab_size
        self.num_attention_heads = num_attention_heads
        self.num_labels = num_labels
        self.seq_len = seq_len
        self.problem_type = problem_type
        self.type_vocab_size = type_vocab_size
        self.time_pos_type = time_pos_type


class TabStaticFormerEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % 2 == 0, f"Cannot use sin/cos positional embeddings with odd dim (got dim={config.hidden_size})"

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

        self.register_buffer("position_embeddings_holder",
                             torch.zeros(1, config.max_position_embeddings, config.hidden_size))
        self.register_buffer("even_div_term", torch.exp(
            2 * torch.arange(0, config.hidden_size, 2) * -(math.log(10000.0) / config.hidden_size)))
        self.register_buffer("odd_div_term", torch.exp(
            2 * torch.arange(1, config.hidden_size, 2) * -(math.log(10000.0) / config.hidden_size)))

    def time_aware_position_embeddings(self, position_embeddings_holder, position_ids_expand):
        position_embeddings_holder[:, :, 0::2] = torch.sin(position_ids_expand * self.even_div_term)
        position_embeddings_holder[:, :, 1::2] = torch.cos(position_ids_expand * self.odd_div_term)
        return position_embeddings_holder

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        hidden_size = self.position_embeddings_holder.shape[-1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]
            position_embeddings_holder = self.position_embeddings_holder[:,
                                         past_key_values_length: seq_length + past_key_values_length, :]
            position_embeddings_holder = position_embeddings_holder.expand(input_shape[0], -1, -1).clone()

        else:
            position_embeddings_holder = self.position_embeddings_holder[:, 0: seq_length, :].expand(input_shape[0], -1,
                                                                                                     -1).clone()

        position_ids_expand = position_ids.view(position_ids.shape + (1,)).expand(
            position_ids.shape + (hidden_size // 2,))

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.time_aware_position_embeddings(position_embeddings_holder, position_ids_expand)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class GetTimeAwarePositionIds(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_a = nn.Parameter(torch.tensor(1.))
        self.time_a = nn.Parameter(torch.tensor(1.))
        self.pi_b = nn.Parameter(torch.tensor(0.))

    def forward(self, position_ids, time_ids):
        return self.pos_a * position_ids + self.time_a * time_ids + self.pi_b


class TabStaticTimePosFormerBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        BertPreTrainedModel.__init__(self, config)
        self.config = config

        self.embeddings = TabStaticFormerEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()


class TabStaticTimePosFormerBertForMaskedLM(BertForMaskedLM):

    def __init__(self, config, vocab):
        BertPreTrainedModel.__init__(self, config)
        self.vocab = vocab
        if self.config.time_pos_type == 'time_aware_sin_cos_position':
            self.time_aware_position_ids = GetTimeAwarePositionIds()
            self.bert = TabStaticTimePosFormerBertModel(config, add_pooling_layer=False)
        elif self.config.time_pos_type == 'sin_cos_position':
            self.bert = TabStaticTimePosFormerBertModel(config, add_pooling_layer=False)
        elif self.config.time_pos_type == 'regular_position' or self.config.time_pos_type is None:
            self.bert = BertModel(config, add_pooling_layer=False)

        self.static_proj = nn.Linear(config.hidden_size, config.field_hidden_size * config.static_ncols)
        self.cls = TabFormerBertOnlyMLMHead(config)
        self.static_cls = TabFormerBertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            time_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            masked_lm_static_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
    ):

        if self.config.time_pos_type == 'time_aware_sin_cos_position':
            position_ids = self.time_aware_position_ids(position_ids, time_ids)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]  # [bsz * seqlen + 1 * hidden]
        sequence_static_output = sequence_output[:, 0:1, :]
        sequence_static_output = self.static_proj(sequence_static_output)
        sequence_dynamic_output = sequence_output[:, 1:, :]

        if not self.config.flatten:
            output_sz = list(sequence_dynamic_output.size())
            expected_static_sz = [output_sz[0], 1 * self.config.static_ncols, -1]
            sequence_static_output = sequence_static_output.view(expected_static_sz)
            expected_dynamic_sz = [output_sz[0], output_sz[1] * self.config.ncols, -1]
            sequence_dynamic_output = sequence_dynamic_output.view(expected_dynamic_sz)
            masked_lm_labels = masked_lm_labels.view(expected_dynamic_sz[0], -1)
            masked_lm_static_labels = masked_lm_static_labels.view(expected_dynamic_sz[0], -1)

        # [bsz * seqlen * vocab_sz]
        prediction_dynamic_scores = self.cls(sequence_dynamic_output)
        prediction_static_scores = self.static_cls(sequence_static_output)

        prediction_scores = torch.cat((prediction_static_scores, prediction_dynamic_scores), dim=1)
        outputs = (prediction_scores,) + outputs[2:]

        # prediction_scores : [bsz x seqlen x vsz]
        # masked_lm_labels  : [bsz x seqlen]

        total_masked_lm_loss = 0

        static_field_names = self.vocab.get_static_field_keys(
            remove_target=True, ignore_special=False)
        dynamic_field_names = self.vocab.get_dynamic_field_keys(
            remove_target=True, ignore_special=False)

        seq_len = prediction_dynamic_scores.size(1)
        for field_idx, field_name in enumerate(dynamic_field_names):
            col_ids = list(range(field_idx, seq_len, len(dynamic_field_names)))

            global_ids_field = self.vocab.get_field_ids(field_name)

            # bsz * 10 * K
            prediction_scores_field = prediction_dynamic_scores[:,
                                      col_ids, :][:, :, global_ids_field]
            masked_lm_labels_field = masked_lm_labels[:, col_ids]
            masked_lm_labels_field_local = self.vocab.get_from_global_ids(global_ids=masked_lm_labels_field,
                                                                          what_to_get='local_ids')

            # when all labels are -100, ignore the loss, because the loss is nan
            if (masked_lm_labels_field_local + 100).sum() != 0:
                nfeas = len(global_ids_field)
                loss_fct = self.get_criterion(
                    field_name, nfeas, prediction_scores.device)
                masked_lm_loss_field = loss_fct(prediction_scores_field.view(-1, len(global_ids_field)),
                                                masked_lm_labels_field_local.view(-1))

                total_masked_lm_loss += masked_lm_loss_field

        seq_len = prediction_static_scores.size(1)
        for field_idx, field_name in enumerate(static_field_names):
            col_ids = list(range(field_idx, seq_len, len(static_field_names)))

            global_ids_field = self.vocab.get_field_ids(field_name)

            prediction_scores_field = prediction_static_scores[:,
                                      col_ids, :][:, :, global_ids_field]
            masked_lm_labels_field = masked_lm_static_labels[:, col_ids]
            masked_lm_labels_field_local = self.vocab.get_from_global_ids(global_ids=masked_lm_labels_field,
                                                                          what_to_get='local_ids')

            # when all labels are -100, ignore the loss, because the loss is nan
            if (masked_lm_labels_field_local + 100).sum() != 0:
                nfeas = len(global_ids_field)
                loss_fct = self.get_criterion(
                    field_name, nfeas, prediction_scores.device)
                try:
                    masked_lm_loss_field = loss_fct(prediction_scores_field.view(-1, len(global_ids_field)),
                                                    masked_lm_labels_field_local.view(-1))
                except:
                    print("psf: ", prediction_scores_field)
                    print("mllfl: ", masked_lm_labels_field_local)
                    raise Exception("stop running because of error")

                total_masked_lm_loss += masked_lm_loss_field

        return (total_masked_lm_loss,) + outputs

    def get_criterion(self, fname, vs, device, cutoffs=False, div_value=4.0):

        if fname in self.vocab.adap_sm_cols:
            if not cutoffs:
                cutoffs = [int(vs / 15), 3 * int(vs / 15), 6 * int(vs / 15)]

            criteria = CustomAdaptiveLogSoftmax(
                in_features=vs, n_classes=vs, cutoffs=cutoffs, div_value=div_value)

            return criteria.to(device)
        else:
            return CrossEntropyLoss()


class TabStaticTimePosFormerBertForPretraining(BertForMaskedLM):

    def __init__(self, config, vocab):
        BertPreTrainedModel.__init__(self, config)
        self.vocab = vocab
        if self.config.time_pos_type == 'time_aware_sin_cos_position':
            self.time_aware_position_ids = GetTimeAwarePositionIds()
            self.bert = TabStaticTimePosFormerBertModel(config, add_pooling_layer=False)
        elif self.config.time_pos_type == 'sin_cos_position':
            self.bert = TabStaticTimePosFormerBertModel(config, add_pooling_layer=False)
        elif self.config.time_pos_type == 'regular_position' or self.config.time_pos_type is None:
            self.bert = BertModel(config, add_pooling_layer=False)
        self.bert = TabStaticTimePosFormerBertModel(config, add_pooling_layer=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            time_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            masked_lm_static_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
            labels=None,
    ):
        if self.config.time_pos_type == 'time_aware_sin_cos_position':
            position_ids = self.time_aware_position_ids(position_ids, time_ids)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]  # [bsz * seqlen * hidden]

        return sequence_output


class TabStaticTimePosFormerBertForClassification(BertForSequenceClassification):

    def __init__(self, config, vocab):
        BertPreTrainedModel.__init__(self, config)
        self.vocab = vocab
        self.ncols = config.ncols
        self.num_labels = config.num_labels
        self.seq_len = config.seq_len
        if self.config.time_pos_type == 'time_aware_sin_cos_position':
            self.time_aware_position_ids = GetTimeAwarePositionIds()
            self.bert = TabStaticTimePosFormerBertModel(config, add_pooling_layer=False)
        elif self.config.time_pos_type == 'sin_cos_position':
            self.bert = TabStaticTimePosFormerBertModel(config, add_pooling_layer=False)
        elif self.config.time_pos_type == 'regular_position' or self.config.time_pos_type is None:
            self.bert = BertModel(config, add_pooling_layer=False)
        self.bert = TabStaticTimePosFormerBertModel(config, add_pooling_layer=False)

        # Initialize weights and apply final processing
        if self.config.flatten:
            self.classifier = nn.Linear(
                (config.ncols * (config.seq_len + 1) * config.hidden_size), config.num_labels)
        else:
            self.classifier = nn.Linear(
                ((config.seq_len + 1) * config.hidden_size), config.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            time_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            masked_lm_labels=None,
            masked_lm_static_labels=None,
            lm_labels=None,
    ):
        if self.config.time_pos_type == 'time_aware_sin_cos_position':
            position_ids = self.time_aware_position_ids(position_ids, time_ids)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]  # [bsz * seqlen * hidden]

        sequence_output = sequence_output.reshape(sequence_output.shape[0], -1)
        logits = self.classifier(sequence_output)  # [bsz * num_classes]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # print("loss shape", loss.shape)
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class MLP_Classification(BertForSequenceClassification):

    def __init__(self, config, vocab):
        BertPreTrainedModel.__init__(self, config)
        self.vocab = vocab
        self.ncols = config.ncols
        self.num_labels = config.num_labels
        self.seq_len = config.seq_len

        # Initialize weights and apply final processing
        if self.config.flatten:
            self.classifier = nn.Linear(
                (config.ncols * config.seq_len * config.hidden_size), config.num_labels)
        else:
            self.classifier = nn.Linear(
                (config.seq_len * config.hidden_size), config.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            time_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            masked_lm_labels=None,
            masked_lm_static_labels=None,
            lm_labels=None,
    ):
        sequence_output = inputs_embeds  # [bsz * seqlen * hidden]

        sequence_output = sequence_output.reshape(sequence_output.shape[0], -1)
        logits = self.classifier(sequence_output)  # [bsz * num_classes]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # print("loss shape", loss.shape)
        output = (logits,)
        return ((loss,) + output) if loss is not None else output
