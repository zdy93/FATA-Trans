# edited by Dongyu Zhang
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_utils import PreTrainedModel

try:
    from transformers.modeling_bert import ACT2FN
except ModuleNotFoundError:
    from transformers.models.bert.modeling_bert import ACT2FN


class TabRawForClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = self.config.num_labels

        self.cata_embedding = nn.Embedding(self.config.vocab_size, self.config.field_hidden_size,
                                           padding_idx=getattr(config, 'pad_token_id', 0), sparse=False)

        self.num_proj = nn.Linear(1, self.config.field_hidden_size)

        # self.proj_act = ACT2FN[self.config.hidden_act]

        self.classifier = nn.Linear(
            self.config.field_hidden_size * self.config.ncols, self.config.num_labels
        )

        self.dropout = nn.Dropout(p=self.config.cls_dropout_prob)

    def forward(self, input_nums=None, input_catas=None, labels=None):
        input_cata_embeds = self.cata_embedding(input_catas)
        input_nums = input_nums.view(input_nums.shape[0], input_nums.shape[1], 1)
        input_num_feas = self.num_proj(input_nums)
        input_embeds = torch.cat([input_num_feas, input_cata_embeds], dim=1)
        input_embeds = input_embeds.view(input_embeds.shape[0], -1)
        # input_embeds = self.proj_act(input_embeds)
        input_embeds = self.dropout(input_embeds)
        logits = self.classifier(input_embeds)

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

        output = (logits,)
        return ((loss,) + output) if loss is not None else output


class TabEmbeddingsForClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = self.config.num_labels
        if self.config.seq_len is not None:
            self.lin_pool = nn.Linear(self.config.hidden_size * self.config.seq_len, self.config.hidden_size)
        self.classifier = nn.Linear(
            self.config.hidden_size, self.config.num_labels
        )
        self.dropout = nn.Dropout(p=self.config.cls_dropout_prob)

    def forward(self, input_embeds=None, labels=None):
        if self.config.seq_len is not None:
            input_embeds = input_embeds.view(input_embeds.shape[0], -1)
            input_embeds = self.lin_pool(input_embeds)

        input_embeds = self.dropout(input_embeds)
        logits = self.classifier(input_embeds)

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

        output = (logits,)
        return ((loss,) + output) if loss is not None else output


class TabEmbeddingsLSTMForClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = self.config.num_labels
        self.LSTM = nn.LSTM(self.config.hidden_size, self.config.rnn_hs, bidirectional=self.config.rnn_bd)
        if self.config.rnn_bd:
            self.classifier = nn.Linear(
                self.config.rnn_hs * 2, self.config.num_labels
            )
        else:
            self.classifier = nn.Linear(
                self.config.rnn_hs, self.config.num_labels
            )
        self.dropout = nn.Dropout(p=self.config.cls_dropout_prob)

    def forward(self, input_embeds=None, labels=None):
        input_embeds = self.dropout(input_embeds)
        input_embeds = input_embeds.permute(1, 0, 2)
        output, (hn, cn) = self.LSTM(input_embeds)
        output = output.permute(1, 0, 2)
        if self.config.rnn_bd:
            f_last = output[:, -1, :self.config.rnn_hs]
            b_last = output[:, 0, self.config.rnn_hs:]
            output_last = torch.cat((f_last, b_last), 1)
            logits = self.classifier(output_last)
        else:
            logits = self.classifier(output[:, -1, :])
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

        output = (logits,)
        return ((loss,) + output) if loss is not None else output


class TabEmbeddingsStaticLSTMForClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = self.config.num_labels
        self.LSTM = nn.LSTM(self.config.hidden_size, self.config.rnn_hs, bidirectional=self.config.rnn_bd)
        self.static_trans = nn.Linear(
            self.config.hidden_size, self.config.rnn_hs * (1 + int(self.config.rnn_bd))
        )
        if isinstance(config.hidden_act, str):
            self.static_act_fn = ACT2FN[self.config.hidden_act]
            self.final_act_fn = ACT2FN[self.config.hidden_act]
        else:
            self.static_act_fn = config.hidden_act
            self.final_act_fn = config.hidden_act
        self.static_LayerNorm = nn.LayerNorm(self.config.rnn_hs * (1 + int(self.config.rnn_bd)),
                                             eps=config.layer_norm_eps)
        self.final_processor = nn.Linear(
            self.config.rnn_hs * (2 + 2 * int(self.config.rnn_bd)),
            self.config.rnn_hs * (2 + 2 * int(self.config.rnn_bd))
        )
        self.final_LayerNorm = nn.LayerNorm(self.config.rnn_hs * (2 + 2 * int(self.config.rnn_bd)),
                                            eps=config.layer_norm_eps)
        self.classifier = nn.Linear(
            self.config.rnn_hs * (2 + 2 * int(self.config.rnn_bd)), self.config.num_labels
        )

        self.dropout = nn.Dropout(p=self.config.cls_dropout_prob)

    def forward(self, input_embeds=None, labels=None):
        input_embeds = self.dropout(input_embeds)
        static_embeds = input_embeds[:, 0, :]
        dynamic_embeds = input_embeds[:, 1:, :]
        dynamic_embeds = dynamic_embeds.permute(1, 0, 2)
        output, (hn, cn) = self.LSTM(dynamic_embeds)
        output = output.permute(1, 0, 2)
        if self.config.rnn_bd:
            f_last = output[:, -1, :self.config.rnn_hs]
            b_last = output[:, 0, self.config.rnn_hs:]
            output_last = torch.cat((f_last, b_last), 1)
        else:
            output_last = output[:, -1, :]
        static_embeds = self.static_trans(static_embeds)
        static_embeds = self.static_act_fn(static_embeds)
        static_embeds = self.static_LayerNorm(static_embeds)
        output_all = torch.cat([static_embeds, output_last], dim=1)
        output_all = self.final_processor(output_all)
        output_all = self.final_act_fn(output_all)
        output_all = self.final_LayerNorm(output_all)
        logits = self.classifier(output_all)
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

        output = (logits,)
        return ((loss,) + output) if loss is not None else output
