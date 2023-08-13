# edited by Dongyu Zhang
from typing import Dict, List, Tuple, Union, Optional, Any
import torch
from dataset.datacollator import TransDataCollatorForLanguageModeling
from transformers.data.data_collator import _torch_collate_batch


class MaskingLabelTransDataCollatorForLanguageModeling(TransDataCollatorForLanguageModeling):

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # batch = self._tensorize_batch(examples)
        example = [i[0] for i in examples]

        batch = _torch_collate_batch(example, self.tokenizer)
        sz = batch.shape

        if self.mlm:
            batch = batch.view(sz[0], -1)
            inputs, labels = self.mask_tokens(batch)
            # print("MLM label shape: ", labels.view(sz).shape)
            # print("MLM batch shape: ", inputs.view(sz).shape)
            return {"input_ids": inputs.view(sz), "masked_lm_labels": labels.view(sz)}
        else:
            inputs, labels = self.mask_label_tokens(batch)
            return {"input_ids": inputs, "labels": labels}

    def mask_label_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for classification task,
        the first token in the last transaction is the label.
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove "
                "the --mlm flag if you want to use this tokenizer. "
            )
        labels = torch.fill_(inputs.clone().detach(), -100)
        pred_label = inputs[:, -1, 0]
        inputs[:, -1, 0] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        labels[:, -1, 0] = pred_label  # We only compute loss on masked tokens
        return inputs, labels


class MaskingLabelTransWithTimePosDataCollatorForLanguageModeling(MaskingLabelTransDataCollatorForLanguageModeling):

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        example = [i[0] for i in examples]
        time_example = [i[1] for i in examples]
        pos_ids_example = [i[2] for i in examples]
        batch = _torch_collate_batch(example, self.tokenizer)
        time_batch = _torch_collate_batch(time_example, self.tokenizer)
        pos_ids_batch = _torch_collate_batch(pos_ids_example, self.tokenizer)
        sz = batch.shape

        if self.mlm:
            batch = batch.view(sz[0], -1)
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs.view(sz), "masked_lm_labels": labels.view(sz),
                    "time_ids": time_batch, "position_ids": pos_ids_batch}
        else:
            inputs, labels = self.mask_label_tokens(batch)
            return {"input_ids": inputs, "labels": labels,
                    "time_ids": time_batch, "position_ids": pos_ids_batch}


class MaskingLabelTransWithStaticDataCollatorForLanguageModeling(MaskingLabelTransDataCollatorForLanguageModeling):

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        dynamic_example = [i[0] for i in examples]
        static_example = [i[1] for i in examples]
        pos_ids_example = [i[2] for i in examples]
        type_ids_example = [i[3] for i in examples]
        dynamic_batch = _torch_collate_batch(dynamic_example, self.tokenizer)
        static_batch = _torch_collate_batch(static_example, self.tokenizer)
        pos_ids_batch = _torch_collate_batch(pos_ids_example, self.tokenizer)
        type_ids_batch = _torch_collate_batch(type_ids_example, self.tokenizer)
        dsz = dynamic_batch.shape
        ssz = static_batch.shape

        if self.mlm:
            dynamic_batch = dynamic_batch.view(dsz[0], -1)
            dynamic_inputs, dynamic_labels = self.mask_tokens(dynamic_batch)
            # static_batch = static_batch.view(ssz[0], -1)
            static_inputs, static_labels = self.mask_tokens(static_batch)
            return {"input_ids": dynamic_inputs.view(dsz), "masked_lm_labels": dynamic_labels.view(dsz),
                    "static_input_ids": static_inputs.view(ssz[0], 1, ssz[-1]),
                    "masked_lm_static_labels": static_labels.view(ssz[0], 1, ssz[-1]),
                    "position_ids": pos_ids_batch, "token_type_ids": type_ids_batch}
        else:
            static_labels = torch.fill_(static_batch.clone().detach(), -100)
            dynamic_inputs, dynamic_labels = self.mask_label_tokens(dynamic_batch)
            return {"input_ids": dynamic_inputs.view(dsz), "masked_lm_labels": dynamic_labels.view(dsz),
                    "static_input_ids": static_batch.view(ssz[0], 1, ssz[-1]),
                    "masked_lm_static_labels": static_labels.view(ssz[0], 1, ssz[-1]),
                    "position_ids": pos_ids_batch, "token_type_ids": type_ids_batch}


class MaskingLabelTransWithStaticAndTimePosDataCollatorForLanguageModeling(MaskingLabelTransDataCollatorForLanguageModeling):

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        dynamic_example = [i[0] for i in examples]
        static_example = [i[1] for i in examples]
        time_example = [i[2] for i in examples]
        pos_ids_example = [i[3] for i in examples]
        type_ids_example = [i[4] for i in examples]
        dynamic_batch = _torch_collate_batch(dynamic_example, self.tokenizer)
        static_batch = _torch_collate_batch(static_example, self.tokenizer)
        time_batch = _torch_collate_batch(time_example, self.tokenizer)
        pos_ids_batch = _torch_collate_batch(pos_ids_example, self.tokenizer)
        type_ids_batch = _torch_collate_batch(type_ids_example, self.tokenizer)
        dsz = dynamic_batch.shape
        ssz = static_batch.shape

        if self.mlm:
            dynamic_batch = dynamic_batch.view(dsz[0], -1)
            dynamic_inputs, dynamic_labels = self.mask_tokens(dynamic_batch)
            # static_batch = static_batch.view(ssz[0], -1)
            static_inputs, static_labels = self.mask_tokens(static_batch)
            return {"input_ids": dynamic_inputs.view(dsz), "masked_lm_labels": dynamic_labels.view(dsz),
                    "static_input_ids": static_inputs.view(ssz[0], 1, ssz[-1]),
                    "masked_lm_static_labels": static_labels.view(ssz[0], 1, ssz[-1]),
                    "time_ids": time_batch, "position_ids": pos_ids_batch, "token_type_ids": type_ids_batch}
        else:
            static_labels = torch.fill_(static_batch.clone().detach(), -100)
            dynamic_inputs, dynamic_labels = self.mask_label_tokens(dynamic_batch)
            return {"input_ids": dynamic_inputs.view(dsz), "masked_lm_labels": dynamic_labels.view(dsz),
                    "static_input_ids": static_batch.view(ssz[0], 1, ssz[-1]),
                    "masked_lm_static_labels": static_labels.view(ssz[0], 1, ssz[-1]),
                    "time_ids": time_batch, "position_ids": pos_ids_batch, "token_type_ids": type_ids_batch}


class MaskingLabelTransDataCollatorForClassification(MaskingLabelTransDataCollatorForLanguageModeling):

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        example = [i[0] for i in examples]
        label = [i[1] for i in examples]

        batch = _torch_collate_batch(example, self.tokenizer)
        label_batch = _torch_collate_batch(label, self.tokenizer)
        inputs, _ = self.mask_label_tokens(batch)
        return {"input_ids": inputs, "labels": label_batch}


class MaskingLabelTransWithStaticDataCollatorForClassification(MaskingLabelTransDataCollatorForLanguageModeling):

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        dynamic_example = [i[0] for i in examples]
        static_example = [i[1] for i in examples]
        pos_ids_example = [i[2] for i in examples]
        type_ids_example = [i[3] for i in examples]
        label = [i[4] for i in examples]
        dynamic_batch = _torch_collate_batch(dynamic_example, self.tokenizer)
        static_batch = _torch_collate_batch(static_example, self.tokenizer)
        pos_ids_batch = _torch_collate_batch(pos_ids_example, self.tokenizer)
        type_ids_batch = _torch_collate_batch(type_ids_example, self.tokenizer)
        label_batch = _torch_collate_batch(label, self.tokenizer)
        dsz = dynamic_batch.shape
        ssz = static_batch.shape

        static_labels = torch.fill_(static_batch.clone().detach(), -100)
        dynamic_inputs, dynamic_labels = self.mask_label_tokens(dynamic_batch)
        return {"input_ids": dynamic_inputs.view(dsz), "masked_lm_labels": dynamic_labels.view(dsz),
                "static_input_ids": static_batch.view(ssz[0], 1, ssz[-1]),
                "masked_lm_static_labels": static_labels.view(ssz[0], 1, ssz[-1]),
                "position_ids": pos_ids_batch, "token_type_ids": type_ids_batch,
                "labels": label_batch}


class MaskingLabelTransWithTimePosDataCollatorForClassification(MaskingLabelTransDataCollatorForLanguageModeling):
    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        example = [i[0] for i in examples]
        time_example = [i[1] for i in examples]
        pos_ids_example = [i[2] for i in examples]
        label = [i[3] for i in examples]
        batch = _torch_collate_batch(example, self.tokenizer)
        time_batch = _torch_collate_batch(time_example, self.tokenizer)
        pos_ids_batch = _torch_collate_batch(pos_ids_example, self.tokenizer)
        label_batch = _torch_collate_batch(label, self.tokenizer)
        sz = batch.shape

        inputs, _ = self.mask_label_tokens(batch)
        return {"input_ids": inputs, "time_ids": time_batch, "position_ids": pos_ids_batch, "labels": label_batch}

class MaskingLabelTransWithStaticAndTimePosDataCollatorForClassification(
      MaskingLabelTransDataCollatorForLanguageModeling):

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        dynamic_example = [i[0] for i in examples]
        static_example = [i[1] for i in examples]
        time_example = [i[2] for i in examples]
        pos_ids_example = [i[3] for i in examples]
        type_ids_example = [i[4] for i in examples]
        label = [i[5] for i in examples]
        dynamic_batch = _torch_collate_batch(dynamic_example, self.tokenizer)
        static_batch = _torch_collate_batch(static_example, self.tokenizer)
        time_batch = _torch_collate_batch(time_example, self.tokenizer)
        pos_ids_batch = _torch_collate_batch(pos_ids_example, self.tokenizer)
        type_ids_batch = _torch_collate_batch(type_ids_example, self.tokenizer)
        label_batch = _torch_collate_batch(label, self.tokenizer)
        dsz = dynamic_batch.shape
        ssz = static_batch.shape

        static_labels = torch.fill_(static_batch.clone().detach(), -100)
        dynamic_inputs, dynamic_labels = self.mask_label_tokens(dynamic_batch)
        return {"input_ids": dynamic_inputs.view(dsz), "masked_lm_labels": dynamic_labels.view(dsz),
                "static_input_ids": static_batch.view(ssz[0], 1, ssz[-1]),
                "masked_lm_static_labels": static_labels.view(ssz[0], 1, ssz[-1]),
                "time_ids": time_batch, "position_ids": pos_ids_batch, "token_type_ids": type_ids_batch,
                "labels": label_batch}


class MaskingLabelTransWithStaticAndTimePosDataCollatorForExtraction(MaskingLabelTransDataCollatorForLanguageModeling):

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        dynamic_example = [i[0] for i in examples]
        static_example = [i[1] for i in examples]
        time_example = [i[2] for i in examples]
        pos_ids_example = [i[3] for i in examples]
        type_ids_example = [i[4] for i in examples]
        dynamic_batch = _torch_collate_batch(dynamic_example, self.tokenizer)
        if self.mlm:
            dynamic_batch, _ = self.mask_label_tokens(dynamic_batch)
        static_batch = _torch_collate_batch(static_example, self.tokenizer)
        time_batch = _torch_collate_batch(time_example, self.tokenizer)
        pos_ids_batch = _torch_collate_batch(pos_ids_example, self.tokenizer)
        type_ids_batch = _torch_collate_batch(type_ids_example, self.tokenizer)
        ssz = static_batch.shape

        return {"input_ids": dynamic_batch, "static_input_ids": static_batch.view(ssz[0], 1, ssz[-1]),
                "time_ids": time_batch, "position_ids": pos_ids_batch, "token_type_ids": type_ids_batch}


class MaskingLabelTransWithStaticDataCollatorForExtraction(MaskingLabelTransDataCollatorForLanguageModeling):

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        dynamic_example = [i[0] for i in examples]
        static_example = [i[1] for i in examples]
        pos_ids_example = [i[2] for i in examples]
        type_ids_example = [i[3] for i in examples]
        dynamic_batch = _torch_collate_batch(dynamic_example, self.tokenizer)
        if self.mlm:
            dynamic_batch, _ = self.mask_label_tokens(dynamic_batch)
        static_batch = _torch_collate_batch(static_example, self.tokenizer)
        pos_ids_batch = _torch_collate_batch(pos_ids_example, self.tokenizer)
        type_ids_batch = _torch_collate_batch(type_ids_example, self.tokenizer)
        ssz = static_batch.shape

        return {"input_ids": dynamic_batch, "static_input_ids": static_batch.view(ssz[0], 1, ssz[-1]),
                "position_ids": pos_ids_batch, "token_type_ids": type_ids_batch}


class MaskingLabelTransWithTimePosDataCollatorForExtraction(MaskingLabelTransDataCollatorForLanguageModeling):

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        example = [i[0] for i in examples]
        time_example = [i[1] for i in examples]
        pos_ids_example = [i[2] for i in examples]
        batch = _torch_collate_batch(example, self.tokenizer)
        if self.mlm:
            batch, _ = self.mask_label_tokens(batch)
        time_batch = _torch_collate_batch(time_example, self.tokenizer)
        pos_ids_batch = _torch_collate_batch(pos_ids_example, self.tokenizer)

        return {"input_ids": batch, "time_ids": time_batch, "position_ids": pos_ids_batch}


class MaskingLabelTransDataCollatorForExtraction(MaskingLabelTransDataCollatorForLanguageModeling):

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # batch = self._tensorize_batch(features)
        example = [i[0] for i in examples]
        batch = _torch_collate_batch(example, self.tokenizer)
        if self.mlm:
            batch, _ = self.mask_label_tokens(batch)
        # print("batch shape: ", batch.shape)
        return {"input_ids": batch}