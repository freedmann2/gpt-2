
set -ex
#export TF_CPP_MIN_VLOG_LEVEL=8

export TF_MLC_LOGGING=1
export TF_CPP_MIN_VLOG_LEVEL=1

export TF_MLC_LOGGING=0
export TF_CPP_MIN_VLOG_LEVEL=0

#exec python3pdb train.py --fresh_model --dataset train.py --sample_ctx 64 --n_embd 128 --n_head 4 --n_layer 4 "$@"
#exec python3pdb train.py --fresh_model --dataset train.py --sample_ctx 128 --n_embd 768 --n_head 12 --n_layer 12 "$@"
#exec python3pdb train.py --fresh_model --dataset train.py --sample_ctx 32 --n_embd 252 --n_head 12 --n_layer 12 --learning_rate 0.0005 --sample_every 999999 --allow_growth --disable_layout_optimizer "$@"
#exec python3pdb train.py --fresh_model --dataset train.py --sample_ctx 32 --n_embd 128 --n_head 4 --n_layer 4 --learning_rate 0.0005 --sample_every 999999 --allow_growth --disable_layout_optimizer "$@"
exec python3pdb train.py --dataset train.py --sample_ctx 32 --n_embd 128 --n_head 4 --n_layer 4 --learning_rate 0.0005 --sample_every 999999 --allow_growth --disable_layout_optimizer "$@"
