from typing import Dict, List, Tuple, Union, Optional, Any
import torch
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers.data.data_collator import _torch_collate_batch


class TransDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # batch = self._tensorize_batch(examples)
        batch = _torch_collate_batch(examples, self.tokenizer)
        sz = batch.shape
        if self.mlm:
            batch = batch.view(sz[0], -1)
            inputs, labels = self.mask_tokens(batch)
            # print("MLM label shape: ", labels.view(sz).shape)
            # print("MLM batch shape: ", inputs.view(sz).shape)
            return {"input_ids": inputs.view(sz), "masked_lm_labels": labels.view(sz)}
        else:
            labels = batch.clone().detach()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": batch, "labels": labels}

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove "
                "the --mlm flag if you want to use this tokenizer. "
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability
        # defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class TransWithTimePosDataCollatorForLanguageModeling(TransDataCollatorForLanguageModeling):

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
            labels = batch.clone().detach()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": batch, "labels": labels,
                    "time_ids": time_batch, "position_ids": pos_ids_batch, }


class TransWithStaticDataCollatorForLanguageModeling(TransDataCollatorForLanguageModeling):

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
            static_labels = static_batch.clone().detach()
            dynamic_labels = dynamic_batch.clone().detach()
            if self.tokenizer.pad_token_id is not None:
                static_labels[static_labels == self.tokenizer.pad_token_id] = -100
                dynamic_labels[dynamic_labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": dynamic_batch.view(dsz), "masked_lm_labels": dynamic_labels.view(dsz),
                    "static_input_ids": static_batch.view(ssz[0], 1, ssz[-1]),
                    "masked_lm_static_labels": static_labels.view(ssz[0], 1, ssz[-1]),
                    "position_ids": pos_ids_batch, "token_type_ids": type_ids_batch}


class TransWithStaticAndTimePosDataCollatorForLanguageModeling(TransDataCollatorForLanguageModeling):

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
            static_labels = static_batch.clone().detach()
            dynamic_labels = dynamic_batch.clone().detach()
            if self.tokenizer.pad_token_id is not None:
                static_labels[static_labels == self.tokenizer.pad_token_id] = -100
                dynamic_labels[dynamic_labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": dynamic_batch.view(dsz), "masked_lm_labels": dynamic_labels.view(dsz),
                    "static_input_ids": static_batch.view(ssz[0], 1, ssz[-1]),
                    "masked_lm_static_labels": static_labels.view(ssz[0], 1, ssz[-1]),
                    "time_ids": time_batch, "position_ids": pos_ids_batch, "token_type_ids": type_ids_batch}


class TransWithStaticAndTimePosDataCollatorForExtraction(TransDataCollatorForLanguageModeling):

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
        ssz = static_batch.shape

        return {"input_ids": dynamic_batch, "static_input_ids": static_batch.view(ssz[0], 1, ssz[-1]),
                "time_ids": time_batch, "position_ids": pos_ids_batch, "token_type_ids": type_ids_batch}


class TransWithStaticDataCollatorForExtraction(TransDataCollatorForLanguageModeling):

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
        ssz = static_batch.shape

        return {"input_ids": dynamic_batch, "static_input_ids": static_batch.view(ssz[0], 1, ssz[-1]),
                "position_ids": pos_ids_batch, "token_type_ids": type_ids_batch}


class TransWithTimePosDataCollatorForExtraction(TransDataCollatorForLanguageModeling):

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        example = [i[0] for i in examples]
        time_example = [i[1] for i in examples]
        pos_ids_example = [i[2] for i in examples]
        batch = _torch_collate_batch(example, self.tokenizer)
        time_batch = _torch_collate_batch(time_example, self.tokenizer)
        pos_ids_batch = _torch_collate_batch(pos_ids_example, self.tokenizer)

        return {"input_ids": batch, "time_ids": time_batch, "position_ids": pos_ids_batch}


class TransDataCollatorForExtraction(DataCollatorForLanguageModeling):

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # batch = self._tensorize_batch(features)
        batch = _torch_collate_batch(features, self.tokenizer)
        # print("batch shape: ", batch.shape)
        return {"input_ids": batch}


class TransDataCollatorForClassification(TransDataCollatorForLanguageModeling):
    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # batch = self.tokenizer.pad(
        #     features,
        #     padding=self.padding,
        #     max_length=self.max_length,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        # )
        # print("features:", features)
        example = [i[0] for i in examples]
        label_example = [i[1] for i in examples]
        # batch = self._tensorize_batch(inputs)
        # labels = self._tensorize_batch(labels)
        batch = _torch_collate_batch(example, self.tokenizer)
        label_batch = _torch_collate_batch(label_example, self.tokenizer)

        return {"input_ids": batch, "labels": label_batch}


class TransWithTimePosDataCollatorForClassification(TransDataCollatorForLanguageModeling):

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        example = [i[0] for i in examples]
        time_example = [i[1] for i in examples]
        pos_ids_example = [i[2] for i in examples]
        label_example = [i[3] for i in examples]
        batch = _torch_collate_batch(example, self.tokenizer)
        time_batch = _torch_collate_batch(time_example, self.tokenizer)
        pos_ids_batch = _torch_collate_batch(pos_ids_example, self.tokenizer)
        label_batch = _torch_collate_batch(label_example, self.tokenizer)

        return {"input_ids": batch, "time_ids": time_batch, "position_ids": pos_ids_batch, "labels": label_batch}


class TransWithStaticDataCollatorForClassification(TransDataCollatorForLanguageModeling):

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        dynamic_example = [i[0] for i in examples]
        static_example = [i[1] for i in examples]
        pos_ids_example = [i[2] for i in examples]
        type_ids_example = [i[3] for i in examples]
        label_example = [i[4] for i in examples]
        dynamic_batch = _torch_collate_batch(dynamic_example, self.tokenizer)
        static_batch = _torch_collate_batch(static_example, self.tokenizer)
        pos_ids_batch = _torch_collate_batch(pos_ids_example, self.tokenizer)
        type_ids_batch = _torch_collate_batch(type_ids_example, self.tokenizer)
        label_batch = _torch_collate_batch(label_example, self.tokenizer)
        ssz = static_batch.shape

        return {"input_ids": dynamic_batch, "static_input_ids": static_batch.view(ssz[0], 1, ssz[-1]),
                "position_ids": pos_ids_batch, "token_type_ids": type_ids_batch, 'labels': label_batch}


class TransWithStaticAndTimePosDataCollatorForClassification(TransDataCollatorForLanguageModeling):

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        dynamic_example = [i[0] for i in examples]
        static_example = [i[1] for i in examples]
        time_example = [i[2] for i in examples]
        pos_ids_example = [i[3] for i in examples]
        type_ids_example = [i[4] for i in examples]
        label_example = [i[5] for i in examples]
        dynamic_batch = _torch_collate_batch(dynamic_example, self.tokenizer)
        static_batch = _torch_collate_batch(static_example, self.tokenizer)
        time_batch = _torch_collate_batch(time_example, self.tokenizer)
        pos_ids_batch = _torch_collate_batch(pos_ids_example, self.tokenizer)
        type_ids_batch = _torch_collate_batch(type_ids_example, self.tokenizer)
        label_batch = _torch_collate_batch(label_example, self.tokenizer)
        ssz = static_batch.shape

        return {"input_ids": dynamic_batch, "static_input_ids": static_batch.view(ssz[0], 1, ssz[-1]),
                "time_ids": time_batch, "position_ids": pos_ids_batch, "token_type_ids": type_ids_batch,
                'labels': label_batch}


class TransEmbedsDataCollatorForClassification(DataCollatorForLanguageModeling):

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.mlm = False
        inputs = [i[0] for i in features]
        labels = [i[1] for i in features]
        # batch = self._tensorize_batch(inputs)
        # labels = self._tensorize_batch(labels)
        inputs = _torch_collate_batch(inputs, self.tokenizer)
        labels = _torch_collate_batch(labels, self.tokenizer)
        # print("label shape: ", labels.shape)
        # print("batch shape: ", batch.shape)
        return {"input_embeds": inputs, "labels": labels}

    def __post_init__(self):
        pass


class TransRawDataCollatorForClassification(DataCollatorForLanguageModeling):

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.mlm = False
        input_nums = [i[0] for i in features]
        input_catas = [i[1] for i in features]
        labels = [i[2] for i in features]
        input_nums = _torch_collate_batch(input_nums, self.tokenizer)
        input_catas = _torch_collate_batch(input_catas, self.tokenizer)
        labels = _torch_collate_batch(labels, self.tokenizer)
        # print("labels shape: ", labels.shape)
        # print("input_nums shape: ", input_nums.shape)
        # print("input_catas shape: ", input_catas.shape)
        return {"input_nums": input_nums, "input_catas": input_catas, "labels": labels}

    def __post_init__(self):
        pass
