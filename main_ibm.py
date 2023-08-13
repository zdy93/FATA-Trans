# edited by Dongyu Zhang
from os import makedirs
from os.path import join, basename
import logging
import numpy as np
import torch
import random
from args import define_new_main_parser
import json

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from dataset.ibm import IBMDataset
from dataset.ibm_time_static import IBMWithTimePosAndStaticSplitDataset
from dataset.ibm_time_pos import IBMWithTimePosDataset
from dataset.ibm_static import IBMWithStaticSplitDataset
from models.modules import TabFormerBertLM, TabFormerBertForClassification, TabFormerBertModel, TabStaticFormerBert, \
    TabStaticFormerBertLM, TabStaticFormerBertClassification
from misc.utils import ordered_split_dataset, compute_cls_metrics, random_split_dataset
from dataset.datacollator import *
import os
os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)
log = logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def main(args):
    # random seeds
    seed = args.seed
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # torch.cuda

    args.vocab_dir = args.output_dir
    # return labels when classification
    args.return_labels = args.cls_task

    dataset_dict = {'IBM': IBMDataset,
                    'IBM_time_pos': IBMWithTimePosDataset,
                    'IBM_static': IBMWithStaticSplitDataset,
                    'IBM_time_static': IBMWithTimePosAndStaticSplitDataset}

    dataset = dataset_dict[args.data_type](cls_task=args.cls_exp_task or args.mlm,
                                           user_ids=args.user_ids,
                                           seq_len=args.seq_len,
                                           root=args.data_root,
                                           fname=args.data_fname,
                                           user_level_cached=args.user_level_cached,
                                           vocab_cached=args.vocab_cached,
                                           external_vocab_path=args.external_vocab_path,
                                           preload_vocab_dir=args.data_root,
                                           save_vocab_dir=args.vocab_dir,
                                           preload_fextension=args.preload_fextension,
                                           fextension=args.fextension,
                                           nrows=args.nrows,
                                           flatten=args.flatten,
                                           stride=args.stride,
                                           return_labels=args.return_labels,
                                           label_category=args.label_category,
                                           pad_seq_first=args.pad_seq_first,
                                           get_rids=args.get_rids,
                                           long_and_sort=args.long_and_sort,
                                           resample_method=args.resample_method,
                                           resample_ratio=args.resample_ratio,
                                           resample_seed=args.resample_seed,
                                           )

    vocab = dataset.vocab
    log.info(f'vocab size: {len(vocab)}')
    custom_special_tokens = vocab.get_special_tokens()

    if args.external_val:
        train_dataset = dataset
        eval_dataset = dataset_dict[args.data_type](cls_task=args.cls_exp_task or args.mlm,
                                                    user_ids=args.user_ids,
                                                    seq_len=args.seq_len,
                                                    root=args.data_root,
                                                    fname=args.data_val_fname,
                                                    user_level_cached=args.user_level_cached,
                                                    vocab_cached=args.vocab_cached,
                                                    external_vocab_path=args.external_vocab_path,
                                                    preload_vocab_dir=args.data_root,
                                                    save_vocab_dir=args.vocab_dir,
                                                    preload_fextension=args.preload_fextension,
                                                    fextension=args.fextension,
                                                    nrows=args.nrows,
                                                    flatten=args.flatten,
                                                    stride=args.stride,
                                                    return_labels=args.return_labels,
                                                    label_category=args.label_category,
                                                    pad_seq_first=False,
                                                    get_rids=args.get_rids,
                                                    long_and_sort=args.long_and_sort,
                                                    resample_method=args.resample_method,
                                                    resample_ratio=args.resample_ratio,
                                                    resample_seed=args.resample_seed,
                                                    )
    else:
        if args.export_task:
            train_dataset = dataset
            eval_dataset = dataset
        else:
            valtrainN = len(dataset)
            trainN = int(0.84 * valtrainN)
            valN = valtrainN - trainN
            lengths = [trainN, valN]
            train_dataset, eval_dataset = random_split_dataset(dataset, lengths)

    test_dataset = dataset_dict[args.data_type](cls_task=args.cls_exp_task or args.mlm,
                                                user_ids=args.user_ids,
                                                seq_len=args.seq_len,
                                                root=args.data_root,
                                                fname=args.data_test_fname,
                                                user_level_cached=args.user_level_cached,
                                                vocab_cached=args.vocab_cached,
                                                external_vocab_path=args.external_vocab_path,
                                                preload_vocab_dir=args.data_root,
                                                save_vocab_dir=args.vocab_dir,
                                                preload_fextension=args.preload_fextension,
                                                fextension=args.fextension,
                                                nrows=args.nrows,
                                                flatten=args.flatten,
                                                stride=1,
                                                return_labels=args.return_labels,
                                                label_category=args.label_category,
                                                pad_seq_first=False,
                                                get_rids=args.get_rids,
                                                long_and_sort=args.long_and_sort,
                                                resample_method=None)

    trainN = len(train_dataset)
    valN = len(eval_dataset)
    testN = len(test_dataset)
    totalN = trainN + valN + testN

    log.info(
        f"# Using external test dataset, lengths: train [{trainN}]  valid [{valN}]  test [{testN}]")
    log.info("# lengths: train [{:.2f}]  valid [{:.2f}]  test [{:.2f}]".format(trainN / totalN, valN / totalN,
                                                                               testN / totalN))

    num_labels = 2
    if args.data_type in ['IBM_time_static', 'IBM_static']:
        if args.mlm:
            tab_net = TabStaticFormerBertLM(custom_special_tokens,
                                            vocab=vocab,
                                            field_ce=args.field_ce,
                                            flatten=args.flatten,
                                            ncols=dataset.ncols,
                                            static_ncols=dataset.static_ncols,
                                            field_hidden_size=args.field_hs,
                                            time_pos_type=args.time_pos_type
                                            )
        elif args.cls_task:
            tab_net = TabStaticFormerBertClassification(custom_special_tokens,
                                                        vocab=vocab,
                                                        field_ce=args.field_ce,
                                                        flatten=args.flatten,
                                                        ncols=dataset.ncols,
                                                        static_ncols=dataset.static_ncols,
                                                        field_hidden_size=args.field_hs,
                                                        seq_len=dataset.seq_len,
                                                        pretrained_dir=args.pretrained_dir,
                                                        num_labels=num_labels,
                                                        time_pos_type=args.time_pos_type,
                                                        problem_type="single_label_classification"
                                                        )
        elif args.export_task:
            tab_net = TabStaticFormerBert(custom_special_tokens,
                                          vocab=vocab,
                                          field_ce=args.field_ce,
                                          flatten=args.flatten,
                                          ncols=dataset.ncols,
                                          static_ncols=dataset.static_ncols,
                                          field_hidden_size=args.field_hs,
                                          seq_len=dataset.seq_len,
                                          pretrained_dir=args.pretrained_dir,
                                          num_labels=num_labels,
                                          time_pos_type=args.time_pos_type
                                          )
    else:
        if args.mlm:
            tab_net = TabFormerBertLM(custom_special_tokens,
                                      vocab=vocab,
                                      field_ce=args.field_ce,
                                      flatten=args.flatten,
                                      ncols=dataset.ncols,
                                      field_hidden_size=args.field_hs,
                                      time_pos_type=args.time_pos_type
                                      )
        elif args.cls_task:
            tab_net = TabFormerBertForClassification(
                custom_special_tokens,
                vocab=vocab,
                field_ce=args.field_ce,
                flatten=args.flatten,
                ncols=dataset.ncols,
                field_hidden_size=args.field_hs,
                seq_len=dataset.seq_len,
                pretrained_dir=args.pretrained_dir,
                num_labels=num_labels,
                problem_type="single_label_classification",
                time_pos_type=args.time_pos_type
            )
        elif args.export_task:
            tab_net = TabFormerBertModel(
                custom_special_tokens,
                vocab=vocab,
                field_ce=args.field_ce,
                flatten=args.flatten,
                ncols=dataset.ncols,
                field_hidden_size=args.field_hs,
                seq_len=dataset.seq_len,
                pretrained_dir=args.pretrained_dir,
                num_labels=num_labels,
                time_pos_type=args.time_pos_type
            )

    log.info(f"model initiated: {tab_net.model.__class__}")

    if args.data_type == "IBM_time_static" and args.mlm:
        collactor_cls = "TransWithStaticAndTimePosDataCollatorForLanguageModeling"
    elif args.data_type == "IBM_time_static" and args.export_task:
        collactor_cls = "TransWithStaticAndTimePosDataCollatorForExtraction"
    elif args.data_type == "IBM_time_static" and args.cls_task:
        collactor_cls = "TransWithStaticAndTimePosDataCollatorForClassification"
    elif args.data_type == "IBM_static" and args.mlm:
        collactor_cls = "TransWithStaticDataCollatorForLanguageModeling"
    elif args.data_type == "IBM_static" and args.export_task:
        collactor_cls = "TransWithStaticDataCollatorForExtraction"
    elif args.data_type == "IBM_static" and args.cls_task:
        collactor_cls = "TransWithStaticDataCollatorForClassification"
    elif args.data_type == "IBM_time_pos" and args.mlm:
        collactor_cls = "TransWithTimePosDataCollatorForLanguageModeling"
    elif args.data_type == "IBM_time_pos" and args.export_task:
        collactor_cls = "TransWithTimePosDataCollatorForExtraction"
    elif args.data_type == "IBM_time_pos" and args.cls_task:
        collactor_cls = "TransWithTimePosDataCollatorForClassification"
    elif args.data_type == "IBM" and args.mlm:
        collactor_cls = "TransDataCollatorForLanguageModeling"
    elif args.data_type == "IBM" and args.export_task:
        collactor_cls = "TransDataCollatorForExtraction"
    elif args.data_type == "IBM" and args.cls_task:
        collactor_cls = "TransDataCollatorForClassification"

    log.info(f"collactor class: {collactor_cls}")
    if args.cls_exp_task:
        data_collator = eval(collactor_cls)(
            tokenizer=tab_net.tokenizer
        )
    else:
        data_collator = eval(collactor_cls)(
            tokenizer=tab_net.tokenizer, mlm=args.mlm, mlm_probability=args.mlm_prob
        )
    if torch.cuda.device_count() > 1:
        per_device_train_batch_size = args.train_batch_size // torch.cuda.device_count()
        per_device_eval_batch_size = args.eval_batch_size // torch.cuda.device_count()
    else:
        per_device_train_batch_size = args.train_batch_size
        per_device_eval_batch_size = args.eval_batch_size

    if args.cls_task or args.export_task:
        label_names = ["labels"]
    else:
        if args.data_type in ["IBM", "IBM_time_pos"]:
            label_names = ["masked_lm_labels"]
        else:
            label_names = ["masked_lm_labels", "masked_lm_static_labels"]
    if args.cls_task:
        metric_for_best_model = 'eval_auc_score'
    else:
        metric_for_best_model = 'eval_loss'
    training_args = TrainingArguments(
        output_dir=args.output_dir,  # output directory
        num_train_epochs=args.num_train_epochs,  # total number of training epochs
        logging_dir=args.log_dir,  # directory for storing logs
        save_steps=args.save_steps,
        do_train=args.do_train,
        do_eval=args.do_eval,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        evaluation_strategy="steps",
        prediction_loss_only=False if args.cls_exp_task else True,
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        eval_steps=args.eval_steps,
        label_names=label_names,
    )

    if args.freeze:
        for name, param in tab_net.model.named_parameters():
            if name.startswith('tb_model.classifier'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    if args.cls_task:
        trainer = Trainer(
            model=tab_net.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_cls_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]

        )
    elif args.export_task:
        trainer = Trainer(
            model=tab_net.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            eval_dataset=train_dataset,
        )
    else:
        trainer = Trainer(
            model=tab_net.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

    if args.export_task:
        if args.nbatches > 1:
            totalN = len(train_dataset)
            bn = args.nbatches
            eachlN = int(totalN / bn)
            reslN = totalN - (bn - 1) * eachlN
            lengths = [eachlN] * (bn - 1)
            lengths.append(reslN)
            batch_data_list = ordered_split_dataset(train_dataset, lengths)
            assert len(train_dataset) == sum([len(s) for s in batch_data_list])
            savez_path = join(args.output_dir, 'all_labels')
            if args.export_last_only:
                np.savez_compressed(savez_path, seq_last_rids=train_dataset.data_seq_last_rids,
                                    seq_last_labels=train_dataset.data_seq_last_labels)
                for ix, batch_data in enumerate(batch_data_list):
                    savez_path = join(args.output_dir, f'batch_{ix}_embeddings')
                    predict_results = trainer.predict(test_dataset=batch_data)
                    if type(predict_results.predictions) is tuple:
                        predictions = predict_results.predictions[1]
                    else:
                        predictions = predict_results.predictions
                    double_full_len = predictions.shape[1]
                    assert double_full_len % 2 == 0
                    full_len = double_full_len // 2
                    np.savez_compressed(savez_path,
                                        seq_last_rids=train_dataset.data_seq_last_rids[batch_data.indices[0]
                                                                                       :batch_data.indices[-1] + 1],
                                        seq_last_labels=train_dataset.data_seq_last_labels[batch_data.indices[0]
                                                                                           :batch_data.indices[-1] + 1],
                                        last_row_embeds=predictions[:, full_len - 1, :],
                                        last_seq_embeds=predictions[:, 2 * full_len - 1, :])
                    print(f"saved file {savez_path}")
                    del predict_results
            else:
                np.savez_compressed(savez_path, sids=train_dataset.data_sids,
                                    seq_last_rids=train_dataset.data_seq_last_rids,
                                    seq_labels=train_dataset.labels)
                for ix, batch_data in enumerate(batch_data_list):
                    savez_path = join(args.output_dir, f'batch_{ix}_embeddings')
                    predict_results = trainer.predict(test_dataset=batch_data)
                    if type(predict_results.predictions) is tuple:
                        predictions = predict_results.predictions[1]
                    else:
                        predictions = predict_results.predictions
                    double_full_len = predictions.shape[1]
                    assert double_full_len % 2 == 0
                    full_len = double_full_len // 2
                    log.info(f"row embeds shape: {predictions[:, :full_len, :].shape}")
                    log.info(f"seq embeds shape: {predictions[:, full_len:, :].shape}")
                    np.savez_compressed(savez_path,
                                        sids=train_dataset.data_sids[batch_data.indices[0]
                                                                     :batch_data.indices[-1] + 1],
                                        seq_last_rids=train_dataset.data_seq_last_rids[batch_data.indices[0]
                                                                                       :batch_data.indices[-1] + 1],
                                        row_embeds=predictions[:, :full_len, :],
                                        seq_embeds=predictions[:, full_len:, :])
                    print(f"saved file {savez_path}")
                    del predict_results
        else:
            predict_results = trainer.predict(test_dataset=train_dataset)
            if type(predict_results.predictions) is tuple:
                predictions = predict_results.predictions[1]
            else:
                predictions = predict_results.predictions
            double_full_len = predictions.shape[1]
            assert double_full_len % 2 == 0
            full_len = double_full_len // 2
            log.info(f"row embeds shape: {predictions[:, :full_len, :].shape}")
            log.info(f"seq embeds shape: {predictions[:, full_len:, :].shape}")
            savez_path = join(args.output_dir, 'all_embeddings')
            if args.export_last_only:
                np.savez_compressed(savez_path, seq_last_rids=train_dataset.data_seq_last_rids,
                                    seq_last_labels=train_dataset.data_seq_last_labels,
                                    last_row_embeds=predictions[:, full_len - 1, :],
                                    last_seq_embeds=predictions[:, 2 * full_len - 1, :])
            else:
                np.savez_compressed(savez_path, sids=train_dataset.data_sids,
                                    seq_last_rids=train_dataset.data_seq_last_rids,
                                    seq_labels=train_dataset.labels,
                                    row_embeds=predictions[:, :full_len, :],
                                    seq_embeds=predictions[:, full_len:, :])

        return

    if args.checkpoint:
        model_path = join(args.output_dir, f'checkpoint-{args.checkpoint}')
        trainer.train(model_path)
    else:
        trainer.train(False)

    # test_preds, test_labels, test_metrics = trainer.predict(test_dataset=test_dataset)
    test_results = trainer.predict(test_dataset=test_dataset)

    args.main_file = basename(__file__)
    performance_dict = vars(args)

    print_str = 'Test Summary: '
    for key, value in test_results.metrics.items():
        performance_dict['test_' + key] = value
        print_str += '{}: {:8f} | '.format(key, value)
    log.info(print_str)

    for key, value in performance_dict.items():
        if type(value) is np.ndarray:
            performance_dict[key] = value.tolist()

    with open(args.record_file, 'a+') as outfile:
        outfile.write(json.dumps(performance_dict) + '\n')

    final_model_path = join(args.output_dir, 'final-model')
    trainer.save_model(final_model_path)
    final_prediction_path = join(args.output_dir, 'prediction_results')
    np.savez_compressed(final_prediction_path,
                        predictions=test_results.predictions, label_ids=test_results.label_ids)


if __name__ == "__main__":

    parser = define_new_main_parser(data_type_choices=["IBM", "IBM_time_pos", "IBM_time_static", "IBM_static"])
    opts = parser.parse_args()

    opts.log_dir = join(opts.output_dir, "logs")
    makedirs(opts.output_dir, exist_ok=True)
    makedirs(opts.log_dir, exist_ok=True)

    file_handler = logging.FileHandler(
        join(opts.log_dir, 'output.log'), 'w', 'utf-8')
    logger.addHandler(file_handler)

    opts.cls_exp_task = opts.cls_task or opts.export_task

    if opts.data_type in ["IBM_time_pos", "IBM_time_static"]:
        assert opts.time_pos_type == 'time_aware_sin_cos_position'
    elif opts.data_type in ["IBM", "IBM_static"]:
        assert opts.time_pos_type in ['sin_cos_position', 'regular_position']

    if opts.mlm and opts.lm_type == "gpt2":
        raise Exception(
            "Error: GPT2 doesn't need '--mlm' option. Please re-run with this flag removed.")

    if (not opts.mlm) and (not opts.cls_exp_task) and opts.lm_type == "bert":
        raise Exception(
            "Error: Bert needs either '--mlm', '--cls_task' or '--export_task' option. Please re-run with this flag "
            "included.")

    main(opts)
