#!/bin/bash

# settings for FATA-Trans
# dt=amazon_time_static
# time_pos_type=time_aware_sin_cos_position
# fextension=static

# settings for TabBERT
dt=amazon
time_pos_type=regular_position
fextension=False

# settings for FATA-Trans (w/o time-aware position embedding)
# dt=amazon_static
# time_pos_type=regular_position
# fextension=static-only

# settings for FATA-Trans (w/o field type aware design)
# dt=amazon_time_pos
# time_pos_type=time_aware_sin_cos_position
# fextension=time-pos

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
pretrained_dir=../IBM_experiment/${data}/${dt}/pretraining/final-model
external_val=True
resample_method=5000
resample_ratio=1
resample_seed=200


if [ "$resample_method" = "None" ]; then
  output_file=out_run_main_amazon_export_${data}_${dt}_${resample_method}_${resample_seed}_${seed}.ipynb
  output_dir=../IBM_experiment/${data}/${resample_method}/${dt}/export
else
  output_file=out_run_main_amazon_export_${data}_${dt}_${resample_method}_${resample_ratio}_${resample_seed}_${seed}.ipynb
  output_dir=../IBM_experiment/${data}/${resample_method}/${resample_ratio}/${dt}/export
fi


papermill \
run_main_amazon_export.ipynb \
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
-p pretrained_dir ${pretrained_dir} \
-p resample_method ${resample_method} \
-p resample_ratio ${resample_ratio} \
-p resample_seed ${resample_seed}