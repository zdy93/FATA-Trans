#!/bin/bash

# settings for FATA-Trans
# dt=IBM_time_static
# time_pos_type=time_aware_sin_cos_position
# fextension=static
# checkpoint=None

# settings for TabBERT
# dt=IBM
# time_pos_type=regular_position
# fextension=False
# checkpoint=None

# settings for FATA-Trans (w/o time-aware position embedding)
# dt=IBM_static
# time_pos_type=regular_position
# fextension=static-only
# checkpoint=None

# settings for FATA-Trans (w/o field type aware design)
dt=IBM_time_pos
time_pos_type=time_aware_sin_cos_position
fextension=time-pos
checkpoint=None

# settings for synthetic transaction
data=credit_card_v2
fname=card_transaction_train
val_fname=card_transaction_val
test_fname=card_transaction_test

# other settings
bs=64
save_steps=5000
eval_steps=5000
resample_method=None
resample_ratio=10
resample_seed=100
external_val=False

if [ "$resample_method" = "None" ]; then
  if [ "$checkpoint" = "None" ]; then
    output_file=out_run_main_ibm_pretraining_${dt}_${resample_method}.ipynb
  else
    output_file=out_run_main_ibm_continue_pretraining_${dt}_${resample_method}_${checkpoint}.ipynb
  fi
  output_dir=../IBM_experiment/${data}/${resample_method}/${dt}/pretraining
else
  if [ "$checkpoint" = "None" ]; then
    output_file=out_run_main_ibm_pretraining_${dt}_${resample_method}_${resample_ratio}.ipynb
  else
    output_file=out_run_main_ibm_continue_pretraining_${dt}_${resample_method}_${resample_ratio}_${checkpoint}.ipynb
  fi
  output_dir=../IBM_experiment/${data}/${resample_method}/${resample_ratio}/${dt}/pretraining
fi

papermill \
run_main_ibm_pretraining.ipynb \
${output_file} \
-k py3.9-visa \
-p data ${data} \
-p dt ${dt} \
-p bs ${bs} \
-p time_pos_type ${time_pos_type} \
-p fextension ${fextension} \
-p fname ${fname} \
-p val_fname ${val_fname} \
-p test_fname ${test_fname} \
-p save_steps ${save_steps} \
-p eval_steps ${eval_steps} \
-p resample_method ${resample_method} \
-p resample_ratio ${resample_ratio} \
-p resample_seed ${resample_seed} \
-p external_val ${external_val} \
-p output_dir ${output_dir} \
-p checkpoint ${checkpoint}