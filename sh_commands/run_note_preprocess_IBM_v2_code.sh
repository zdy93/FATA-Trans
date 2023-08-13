ty=${1:-2018}
sl=${2:-10}

papermill \
preprocess_IBM_v2.ipynb \
out_preprocess_IBM_v2.ipynb \
-k py3.9-visa \
-p file_path './data/credit_card_v2/card_transaction.v1.csv' \
-p train_path './data/credit_card_v2/card_transaction_train.csv' \
-p test_path './data/credit_card_v2/card_transaction_test.csv' \
-p train_test_thres_year ${ty} \
-p seq_len ${sl} \
-p consider_card False