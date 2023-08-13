fn=${1:-"Movies_and_TV_5"}
sl=${2:-10}

papermill \
preprocess_amazon_liang.ipynb \
out_preprocess_amazon_liang_${fn}.ipynb \
-k py3.7-visa \
-p file_name "amazon/"${fn}".json.gz" \
-p seq_len ${sl}