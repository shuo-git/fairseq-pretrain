# This script is adapted from examples of finetuning BART models on GLUE tasks: https://github.com/shuo-git/fairseq-pretrain/blob/master/examples/bart/README.glue.md

TOTAL_NUM_UPDATES=33112  # 10 epochs through RTE for bsz 16
WARMUP_UPDATES=1986      # 6 percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
MAX_SENTENCES=32        # Batch size.
LM_PATH=/home/gzc/fairseq-test/checkpoints/transformer_wikitext-small-103-raw/checkpoint_best.pt

CUDA_VISIBLE_DEVICES=5 fairseq-train QNLI-bin/ \
    --restore-file $LM_PATH \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --add-prev-output-tokens \
    --layernorm-embedding \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 \
    --arch transformer_lm \
    --tokens-per-sample 512 \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 10 \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric;