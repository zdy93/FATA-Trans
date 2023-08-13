#!/bin/bash

fname=process_IBM_dataset
dfname=card_transaction_train
val_dfname=card_transaction_val
test_dfname=card_transaction_test
root=./data/credit_card_v2/
fextension=time-pos
preload_fextension=preload-test

user_level_cached=False
vocab_cached=False
external_vocab_path=False
external_val=False


papermill \
${fname}.ipynb \
out_${fname}_${fextension}.ipynb \
-k py3.7-visa \
-p root ${root} \
-p fname ${dfname} \
-p val_fname ${val_dfname} \
-p test_fname ${test_dfname} \
-p fextension ${fextension} \
-p preload_fextension ${preload_fextension} \
-p user_level_cached ${user_level_cached} \
-p vocab_cached ${vocab_cached} \
-p external_vocab_path ${external_vocab_path} \
-p external_val ${external_val}