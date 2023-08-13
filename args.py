# edited by Dongyu Zhang
import argparse


def define_main_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("--jid", type=int,
                        default=1,
                        help="job id: 1[default] used for job queue")
    parser.add_argument("--seed", type=int,
                        default=9,
                        help="seed to use: 9[default]")

    parser.add_argument("--lm_type", default='bert', choices=['bert', 'gpt2'],
                        help="gpt or bert choice.")
    parser.add_argument("--flatten", action='store_true',
                        help="enable flattened input, no hierarchical")
    parser.add_argument("--field_ce", action='store_true',
                        help="enable field wise CE")
    parser.add_argument("--mlm", action='store_true',
                        help="masked lm loss; pass it for BERT")
    parser.add_argument("--cls_task", action='store_true',
                        help="classification loss; pass it for BERT")
    parser.add_argument("--reg_task", action='store_true',
                        help="regression loss; pass it for BERT")
    parser.add_argument("--export_task", action='store_true',
                        help='if export embeddings or not')
    parser.add_argument("--export_last_only", action='store_true',
                        help='when export_task is True, if only export last embedding of each sequence or not')
    parser.add_argument("--mlm_prob", type=float,
                        default=0.15,
                        help="mask mlm_probability")
    parser.add_argument('--freeze', action='store_true',
                        help='If set, freezes all layer parameters except for the output layer')

    parser.add_argument("--data_type", type=str,
                        default="card", choices=['card', 'prsa', 'card_cls', 'prsa_reg', 'visa', 'visa_time',
                                                 'visa_time_split', 'visa_split', 'visa_time_pos'],
                        help='dataset type')
    parser.add_argument("--time_pos_type", type=str,
                        default="regular_position",
                        choices=['time_aware_sin_cos_position', 'sin_cos_position', 'regular_position'],
                        help='position embedding type')
    parser.add_argument("--data_root", type=str,
                        default="./data/credit_card/",
                        help='root directory for files')
    parser.add_argument("--data_fname", type=str,
                        default="card_transaction.v2",
                        help='file name of transaction')
    parser.add_argument("--data_extension", type=str,
                        default="",
                        help="file name extension to add to cache")

    parser.add_argument("--oot_test", action='store_true',
                        help="if external test (oot) file will be used or not")
    parser.add_argument("--test_data_root", type=str,
                        default="./data/credit_card/",
                        help='root directory for oot test files')
    parser.add_argument("--test_data_fname", type=str,
                        default="card_transaction.v2",
                        help='file name of oot test transaction')
    parser.add_argument("--test_data_extension", type=str,
                        default="_test",
                        help="file name extension to add to cache of oot test data")

    parser.add_argument("--vocab_file", type=str,
                        default='vocab.nb',
                        help="cached vocab file")
    parser.add_argument('--user_ids', nargs='+',
                        default=None,
                        help='pass list of user ids to filter data by')
    parser.add_argument("--cached", action='store_true',
                        help='use cached data files')
    parser.add_argument("--encoder_cached", action='store_true',
                        help='use cached encoder files')
    parser.add_argument("--vocab_cached", action='store_true',
                        help='use cached vocab files')
    parser.add_argument("--external_encoder_fname", type=str,
                        default="./data/credit_card/preprocessed/card_transaction.v1.encoder_fit.pkl",
                        help='file name of externel encoder')
    parser.add_argument("--external_vocab_fname", type=str,
                        default="./data/credit_card/vocab_ob",
                        help='file name of externel vocab file')
    parser.add_argument("--nrows", type=int,
                        default=None,
                        help="no of transactions to use")
    parser.add_argument("--label_category", type=str,
                        default="last_label", choices=['last_label', 'window_label', 'sequence_label'],
                        help='type of target label used for the classifier')

    parser.add_argument("--nbatches", type=int,
                        default=None,
                        help="no of batches to use in export task")

    parser.add_argument("--record_file", type=str,
                        default='experiments',
                        help="path to experiment record")
    parser.add_argument("--output_dir", type=str,
                        default='checkpoints',
                        help="path to model dump")
    parser.add_argument("--pretrained_dir", type=str,
                        default=None,
                        help="path to load pretrained model")
    parser.add_argument("--vocab_dir", type=str,
                        default=None,
                        help="path to load vocab file")
    parser.add_argument("--checkpoint", type=int,
                        default=0,
                        help='set to continue training from checkpoint')
    parser.add_argument("--do_train", action='store_true',
                        help="enable training flag")
    parser.add_argument("--do_eval", action='store_true',
                        help="enable evaluation flag")
    parser.add_argument("--do_prediction", action='store_true',
                        help="enable prediction flag")
    parser.add_argument("--save_steps", type=int,
                        default=500,
                        help="set checkpointing")
    parser.add_argument("--eval_steps", type=int,
                        default=500,
                        help="Number of update steps between two evaluations")
    parser.add_argument("--num_train_epochs", type=int,
                        default=3,
                        help="number of training epochs")
    parser.add_argument("--train_batch_size", type=int,
                        default=8,
                        help="training batch size")
    parser.add_argument("--eval_batch_size", type=int,
                        default=8,
                        help="eval batch size")
    parser.add_argument("--stride", type=int,
                        default=5,
                        help="stride for transaction sliding window")
    parser.add_argument("--seq_len", type=int,
                        default=10,
                        help="length for transaction sliding window")

    parser.add_argument("--field_hs", type=int,
                        default=768,
                        help="hidden size for transaction transformer")
    parser.add_argument("--skip_user", action='store_true',
                        help="if user field to be skipped or added (default add)")
    parser.add_argument("--pad_seq_first", action='store_true',
                        help="if each user first few transactions will be padded to build sequences (default not pad)")
    parser.add_argument("--get_rids", action='store_true',
                        help="if transaction rid will be stored")
    parser.add_argument("--long_and_sort", action='store_true',
                        help="if transaction data is very and sorted by id, if so, will not use .loc function to process data")

    return parser


def remove_argument(parser, arg_list):
    for arg in arg_list:
        for action in parser._actions:
            opts = action.option_strings
            if (opts and opts[0] == arg) or action.dest == arg:
                parser._remove_action(action)
                break

        for action in parser._action_groups:
            for group_action in action._group_actions:
                if group_action.dest == arg:
                    action._group_actions.remove(group_action)
    return parser


def redefine_argument(parser, arg, new_choices=None, new_help=None, new_default=None):
    for action in parser._actions:
        opts = action.option_strings
        if (opts and opts[0] == arg) or action.dest == arg:
            if new_choices is not None:
                action.choices = new_choices
            if new_help is not None:
                action.help = new_help
            if new_default is not None:
                action.default = new_default
    return parser


def define_new_main_parser(parser=None, data_type_choices=["IBM", "IBM_time_pos", "IBM_time_static", "IBM_static"]):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser = define_main_parser(parser)
    remove_arg_list = ["cached", "encoder_cached", "oot_test", "test_data_root", "test_data_fname",
                       "test_data_extension", "reg_task", "data_extension"]
    parser = remove_argument(parser, remove_arg_list)
    parser = redefine_argument(parser, "data_type", new_choices=data_type_choices)
    parser = redefine_argument(parser, "vocab_cached",
                               new_help='if external final vocab (not the preload vocab) will be used,'
                                        'if so, final vocab will be loaded')
    parser = redefine_argument(parser, "data_fname", new_help='file name of original training data',
                               new_default='card_transaction_train')
    parser.add_argument("--user_level_cached", action='store_true',
                        help='if user level data will be used, if so, cached data will be loaded')
    parser.add_argument("--external_vocab_path", type=str, help='path to the external final vocab file')
    parser.add_argument("--preload_fextension", type=str, help='file name extension used by encoded data and '
                                                               'preloaded vocab files')
    parser.add_argument("--fextension", type=str, default="",
                        help='file name extension used by cached user level data and final vocab file')
    parser.add_argument("--data_val_fname", type=str, default='card_transaction_val',
                        help='file name of original validation data')
    parser.add_argument("--data_test_fname", type=str, default='card_transaction_test',
                        help='file name of original test data')
    parser.add_argument("--resample_method", default=None,
                        help='a method for data resampling. If None, then not resampling, if "upsample",'
                             'then the class with less samples will be upsampled, if "downsample", '
                             'then the class with more samples will be downsampled, if a int is given, '
                             'then each class will be sampled by the given number')
    parser.add_argument("--resample_ratio", type=float, default=10,
                        help='ratio for resample data, resample_ratio = # of negative samples / # of positive samples')
    parser.add_argument("--resample_seed", type=int, default=100,
                        help='random seed for data resample')
    parser.add_argument("--external_val", action='store_true',
                        help='if validation dataset is not a subset of train dataset, if so, an external validation '
                             'dataset will be loaded')
    return parser





def define_embeds_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("--jid", type=int,
                        default=1,
                        help="job id: 1[default] used for job queue")
    parser.add_argument("--seed", type=int,
                        default=9,
                        help="seed to use: 9[default]")
    parser.add_argument("--flatten", action='store_true',
                        help="enable flattened input, no hierarchical")

    parser.add_argument("--single_file", action='store_true',
                        help="if all embeddings stored in a single file or not")
    parser.add_argument("--data_root", type=str,
                        default="./data/credit_card/",
                        help='root directory for files')
    parser.add_argument("--data_type", type=str,
                        default="not-split", choices=['not-split', 'split', 'raw'],
                        help='dataset type')
    parser.add_argument("--single_fname", type=str,
                        default="all_embeddings.npz",
                        help='file name of a single file contains both embeddings and label data')
    parser.add_argument("--label_fname", type=str,
                        default="all_labels.npz",
                        help='file name of label data only')
    parser.add_argument("--batch_fname_prefix", type=str,
                        default="batch_",
                        help='file name prefix of embeddings in batch')
    parser.add_argument("--batch_fname_suffix", type=str,
                        default="_embeddings.npz",
                        help='file name suffix of embeddings in batch')
    parser.add_argument("--data_fname", type=str,
                        default="card_transaction.v2",
                        help='file name of transaction data')
    parser.add_argument("--seq_fname", type=str,
                        default="seq_level_data",
                        help='file name of sequence data')
    parser.add_argument("--data_extension", type=str,
                        default="",
                        help="file name extension to add to cache")

    parser.add_argument("--cached", action='store_true',
                        help='use cached data files')
    parser.add_argument("--encoder_cached", action='store_true',
                        help='use cached encoder files')
    parser.add_argument("--vocab_cached", action='store_true',
                        help='use cached vocab files')
    parser.add_argument("--external_encoder_fname", type=str,
                        default="./data/credit_card/preprocessed/card_transaction.v1.encoder_fit.pkl",
                        help='file name of externel encoder')
    parser.add_argument("--external_vocab_fname", type=str,
                        default="./data/credit_card/vocab_ob",
                        help='file name of externel vocab file')

    parser.add_argument("--standardize", action='store_true',
                        help="standardize raw numerical features flag")

    parser.add_argument("--test_cached", action='store_true',
                        help='use cached oot data files')
    parser.add_argument("--oot_test", action='store_true',
                        help="if external test (oot) file will be used or not")
    parser.add_argument("--test_single_file", action='store_true',
                        help="if all oot embeddings stored in a single file or not")
    parser.add_argument("--test_data_root", type=str,
                        default="./data/credit_card/",
                        help='root directory for oot files')
    parser.add_argument("--test_single_fname", type=str,
                        default="all_embeddings.npz",
                        help='file name of a single file contains both oot embeddings and oot label data')
    parser.add_argument("--test_label_fname", type=str,
                        default="all_labels.npz",
                        help='file name of oot label data only')
    parser.add_argument("--test_batch_fname_prefix", type=str,
                        default="batch_",
                        help='file name prefix of oot embeddings in batch')
    parser.add_argument("--test_batch_fname_suffix", type=str,
                        default="_embeddings.npz",
                        help='file name suffix of oot embeddings in batch')

    parser.add_argument("--test_data_fname", type=str,
                        default="card_transaction.v2",
                        help='file name of oot transaction data')
    parser.add_argument("--test_seq_fname", type=str,
                        default="seq_level_data",
                        help='file name of oot sequence data')
    parser.add_argument("--test_data_extension", type=str,
                        default="_test",
                        help="file name extension to add to cache of oot test data")

    parser.add_argument("--data_category", type=str,
                        default="sequence_last_embedding",
                        choices=['sequence_last_embedding', 'row_last_embedding', 'all_last_embedding',
                                 'sequence_embeddings', 'row_embeddings', 'all_embeddings', 'raw'],
                        help='type of embedding used for the classifier')
    parser.add_argument("--label_category", type=str,
                        default="last_label", choices=['last_label', 'window_label', 'sequence_label'],
                        help='type of target label used for the classifier')
    parser.add_argument("--nbatches", type=int,
                        default=None,
                        help="# of embedding files in batch")
    parser.add_argument("--test_nbatches", type=int,
                        default=None,
                        help="# of oot embedding files in batch")
    parser.add_argument("--ncols", type=int,
                        default=12,
                        help="# of columns in transactions data")
    parser.add_argument("--dynamic_ncols", type=int,
                        default=12,
                        help="# of dynamic columns in transactions data")
    parser.add_argument("--static_ncols", type=int,
                        default=12,
                        help="# of static columns in transactions data")
    parser.add_argument("--seq_len", type=int,
                        default=10,
                        help="length for transaction sliding window")
    parser.add_argument("--field_hs", type=int,
                        default=64,
                        help="size of transaction embedding")
    parser.add_argument("--num_labels", type=int,
                        default=2,
                        help="# of classfication label")
    parser.add_argument("--nrows", type=int,
                        default=None,
                        help="# of transactions to use")
    parser.add_argument("--test_nrows", type=int,
                        default=None,
                        help="no of transactions to use in oot dataset")
    parser.add_argument("--dropout_prob", type=float,
                        default=0.1,
                        help="dropout probability of classifier")
    parser.add_argument("--rnn_hs", type=int,
                        default=None,
                        help="hidden size of LSTM layer")
    parser.add_argument("--rnn_bd", action='store_true',
                        default=None,
                        help="if using bidirectional LSTM layer or not")
    parser.add_argument("--learning_rate", type=float,
                        default=1e-3,
                        help="learning rate of classifier")

    parser.add_argument("--checkpoint", type=int,
                        default=0,
                        help='set to continue training from checkpoint')
    parser.add_argument("--do_train", action='store_true',
                        help="enable training flag")
    parser.add_argument("--do_eval", action='store_true',
                        help="enable evaluation flag")
    parser.add_argument("--do_prediction", action='store_true',
                        help="enable prediction flag")
    parser.add_argument("--save_steps", type=int,
                        default=500,
                        help="set checkpointing")
    parser.add_argument("--eval_steps", type=int,
                        default=500,
                        help="Number of update steps between two evaluations")
    parser.add_argument("--num_train_epochs", type=int,
                        default=3,
                        help="number of training epochs")
    parser.add_argument("--train_batch_size", type=int,
                        default=8,
                        help="training batch size")
    parser.add_argument("--eval_batch_size", type=int,
                        default=8,
                        help="eval batch size")
    parser.add_argument("--record_file", type=str,
                        default='experiments',
                        help="path to experiment record")
    parser.add_argument("--output_dir", type=str,
                        default='checkpoints',
                        help="path to model dump")
    parser.add_argument("--problem_type", type=str,
                        default="single_label_classification",
                        choices=['regression', 'single_label_classification', 'multi_label_classification'],
                        help='type of task')
    parser.add_argument("--embeds_model_type", type=str,
                        default="lstm", choices=['lstm', 'mlp', 'lstm-split'],
                        help='type of model')
    parser.add_argument("--metric_for_best", type=str,
                        default="eval_loss", choices=['eval_auc_score', 'eval_loss'],
                        help='metric for best model')

    return parser
