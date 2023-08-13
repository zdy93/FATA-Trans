#!/bin/bash

fname=process_amazon_dataset

# settings for Amazon Movies and TV
dfname=Movies_and_TV_5_train
val_dfname=Movies_and_TV_5_val
test_dfname=Movies_and_TV_5_test
root=./data/amazon_movie/

# settings for Amazon Electrionics
# dfname=Electronics_5_train
# val_dfname=Electronics_5_val
# test_dfname=Electronics_5_test
# root=./data/amazon_electronics/

# other settings
fextension=static # for FATA-Trans
# fextension=static-only # for FATA-Trans (w/o time-aware position embedding)
# fextension=time-pos # for FATA-Trans (w/o field type aware design)
# fextension=False # for TabBERT
preload_fextension=preload-test
reviewer_level_cached=False
vocab_cached=False
external_vocab_path=False

papermill \
${fname}.ipynb \
out_${fname}_${fextension}.ipynb \
-k py3.9-visa \
-p root ${root} \
-p fname ${dfname} \
-p val_fname ${val_dfname} \
-p test_fname ${test_dfname} \
-p fextension ${fextension} \
-p preload_fextension ${preload_fextension} \
-p reviewer_level_cached ${reviewer_level_cached} \
-p vocab_cached ${vocab_cached} \
-p external_vocab_path ${external_vocab_path}