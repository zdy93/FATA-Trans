#!/bin/bash

# settings for FATA-Trans
# dt=amazon_time_static
# time_pos_type=time_aware_sin_cos_position
# fextension=static
# checkpoint=None

# settings for TabBERT
dt=amazon
time_pos_type=regular_position
fextension=False
checkpoint=None

# settings for FATA-Trans (w/o time-aware position embedding)
# dt=amazon_static
# time_pos_type=regular_position
# fextension=static-only
# checkpoint=None

# settings for FATA-Trans (w/o field type aware design)
# dt=amazon_time_pos
# time_pos_type=time_aware_sin_cos_position
# fextension=time-pos
# checkpoint=None

# settings for Amazon Electronics
data=amazon_electronics
fname=Electronics_5_train
val_fname=Electronics_5_val
test_fname=Electronics_5_test

# settings for Amazon Movie
# data=amazon_movie
# fname=Movies_and_TV_5_train
# val_fname=Movies_and_TV_5_val
# test_fname=Movies_and_TV_5_test

# other settings
bs=64
num_train_epochs=3
save_steps=2500
eval_steps=2500
external_val=True


if [ "$checkpoint" = "None" ]; then
output_file=out_run_main_amazon_trainfromscratch_${dt}.ipynb
else
output_file=out_run_main_amazon_continue_trainfromscratch_${dt}.ipynb
fi
output_dir=../IBM_experiment/${data}/${dt}/trainfromscratch


papermill \
run_main_amazon_trainfromscratch.ipynb \
${output_file} \
-k py3.9-visa \
-p data ${data} \
-p dt ${dt} \
-p bs ${bs} \
-p num_train_epochs ${num_train_epochs} \
-p time_pos_type ${time_pos_type} \
-p fextension ${fextension} \
-p fname ${fname} \
-p val_fname ${val_fname} \
-p test_fname ${test_fname} \
-p save_steps ${save_steps} \
-p eval_steps ${eval_steps} \
-p external_val ${external_val} \
-p output_dir ${output_dir} \
-p checkpoint ${checkpoint}