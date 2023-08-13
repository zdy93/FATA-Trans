#!/bin/bash

fname=preload_dataset

# settings for Amamzon electronics
# dfname=Electronics_5_train
# val_dfname=Electronics_5_val
# test_dfname=Electronics_5_test
# root=./data/amazon_electronics/
# data_source=amazon
# external_val=True


# settings for Amazon movies and TV
# dfname=Movies_and_TV_5_train
# val_dfname=Movies_and_TV_5_val
# test_dfname=Movies_and_TV_5_test
# root=./data/amazon_movie/
# data_source=amazon
# external_val=True

# settings for synthetic transaction 
dfname=card_transaction_train
val_dfname=card_transaction_val
test_dfname=card_transaction_test
root=./data/credit_card_v2/
data_source=IBM
external_val=False

# other setttings
fextension=preload-test
cached=False
val_cached=False
test_cached=False
vocab_cached=False
encoder_cached=False
external_encoder_path=False





papermill \
${fname}.ipynb \
out_${fname}_${fextension}.ipynb \
-k py3.9-visa \
-p root ${root} \
-p fname ${dfname} \
-p val_fname ${val_dfname} \
-p test_fname ${test_dfname} \
-p fextension ${fextension} \
-p cached ${cached} \
-p val_cached ${val_cached} \
-p test_cached ${test_cached} \
-p encoder_cached ${encoder_cached} \
-p vocab_cached ${vocab_cached} \
-p external_encoder_path ${external_encoder_path} \
-p data_source ${data_source} \
-p external_val ${external_val}