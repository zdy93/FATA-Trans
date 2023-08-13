# edited by Dongyu Zhang
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

try:
    from transformers.modeling_bert import ACT2FN, BertLayerNorm
    from transformers.modeling_bert import BertModel, BertPreTrainedModel, BertForMaskedLM, \
        BertForSequenceClassification
    from transformers.configuration_bert import BertConfig
except ModuleNotFoundError:
    from transformers.models.bert.modeling_bert import ACT2FN

    BertLayerNorm = nn.LayerNorm
    from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertForMaskedLM, \
        BertForSequenceClassification
    from transformers.models.bert.configuration_bert import BertConfig

from models.custom_criterion import CustomAdaptiveLogSoftmax
from models.tabformer_static_bert import TabStaticFormerEmbeddings, GetTimeAwarePositionIds

from typing import List, Optional, Tuple, Union


class TabFormerBertConfig(BertConfig):
    def __init__(
            self,
            flatten=True,
            ncols=12,
            vocab_size=30522,
            field_hidden_size=64,
            hidden_size=768,
            num_attention_heads=12,
            pad_token_id=0,
            num_labels=2,
            seq_len=10,
            problem_type=None,
            time_pos_type=None,
            **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.ncols = ncols
        self.field_hidden_size = field_hidden_size
        self.hidden_size = hidden_size
        self.flatten = flatten
        self.vocab_size = vocab_size
        self.num_attention_heads = num_attention_heads
        self.num_labels = num_labels
        self.seq_len = seq_len
        self.problem_type = problem_type
        self.time_pos_type = time_pos_type


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

        # Fake decoder
        # self.dd_decoder = nn.Linear(
        #     config.hidden_size, 2, bias=False)

        # self.dd_bias = nn.Parameter(torch.zeros(2))
        # self.dd_decoder.bias = self.dd_bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        # print(f"1st hidden state shape: {hidden_states.shape}")
        hidden_states = self.decoder(hidden_states)
        # hidden_states = self.dd_decoder(hidden_states)
        # print(f"2nd hidden state shape: {hidden_states.shape}")
        return hidden_states


class TabFormerBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = TabFormerBertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class TabFormerBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config, vocab):
        super().__init__(config)

        self.vocab = vocab
        self.cls = TabFormerBertOnlyMLMHead(config)
        if self.config.time_pos_type == 'time_aware_sin_cos_position':
            self.time_aware_position_ids = GetTimeAwarePositionIds()
            self.bert.embeddings = TabStaticFormerEmbeddings(config)
        elif self.config.time_pos_type == 'sin_cos_position':
            self.bert.embeddings = TabStaticFormerEmbeddings(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
            time_ids=None,
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

        if not self.config.flatten:
            output_sz = list(sequence_output.size())
            expected_sz = [output_sz[0], output_sz[1] * self.config.ncols, -1]
            sequence_output = sequence_output.view(expected_sz)
            masked_lm_labels = masked_lm_labels.view(expected_sz[0], -1)

        # [bsz * seqlen * vocab_sz]
        prediction_scores = self.cls(sequence_output)

        # FAKE loss function computation
        # print(f"prediction_scores shape: {prediction_scores.shape}")
        # target = torch.empty((prediction_scores.shape[0]), dtype=torch.long, device=inputs_embeds.device).random_(2)
        # loss = CrossEntropyLoss()
        # total_masked_lm_loss = 0
        # for i in range(prediction_scores.shape[1]):
        #     total_masked_lm_loss += loss(prediction_scores[:,i,:], target)

        outputs = (prediction_scores,) + outputs[2:]

        # prediction_scores : [bsz x seqlen x vsz]
        # masked_lm_labels  : [bsz x seqlen]

        total_masked_lm_loss = 0

        seq_len = prediction_scores.size(1)
        # TODO : remove_target is True for card
        field_names = self.vocab.get_field_keys(
            remove_target=True, ignore_special=False)
        for field_idx, field_name in enumerate(field_names):
            col_ids = list(range(field_idx, seq_len, len(field_names)))

            global_ids_field = self.vocab.get_field_ids(field_name)

            # bsz * 10 * K
            prediction_scores_field = prediction_scores[:, col_ids, :][:, :, global_ids_field]
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


class TabFormerBertForPretraining(TabFormerBertForMaskedLM):
    def __init__(self, config, vocab):
        super().__init__(config, vocab)

        self.bert = BertModel(config)
        if self.config.time_pos_type == 'time_aware_sin_cos_position':
            self.time_aware_position_ids = GetTimeAwarePositionIds()
            self.bert.embeddings = TabStaticFormerEmbeddings(config)
        elif self.config.time_pos_type == 'sin_cos_position':
            self.bert.embeddings = TabStaticFormerEmbeddings(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
            labels=None,
            time_ids=None,
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

        # if not self.config.flatten:
        #     output_sz = list(sequence_output.size())
        #     expected_sz = [output_sz[0], output_sz[1]*self.config.ncols, -1]
        #     sequence_output = sequence_output.view(expected_sz)

        return sequence_output


class TabFormerBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, vocab):
        super().__init__(config)
        self.ncols = config.ncols
        self.num_labels = config.num_labels
        self.vocab = vocab
        self.seq_len = config.seq_len
        if self.config.time_pos_type == 'time_aware_sin_cos_position':
            self.time_aware_position_ids = GetTimeAwarePositionIds()
            self.bert.embeddings = TabStaticFormerEmbeddings(config)
        elif self.config.time_pos_type == 'sin_cos_position':
            self.bert.embeddings = TabStaticFormerEmbeddings(config)
        if self.config.flatten:
            self.classifier = nn.Linear(
                (config.ncols * config.seq_len * config.hidden_size), config.num_labels)
        else:
            self.classifier = nn.Linear(
                (config.seq_len * config.hidden_size), config.num_labels)

        # Initialize weights and apply final processing
        self.init_weights()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            time_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        if self.config.time_pos_type == 'time_aware_sin_cos_position':
            position_ids = self.time_aware_position_ids(position_ids, time_ids)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
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

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
