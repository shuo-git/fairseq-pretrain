# Follow the FAIRSEQ examples of training language models (especially data downloading and preprocessing): https://github.com/shuo-git/fairseq-pretrain/blob/master/examples/language_model/README.md

export CUDA_VISIBLE_DEVICES=3
fairseq-train --task language_modeling \
  data-bin/wikitext-small-103-raw \
  --save-dir checkpoints/transformer_wikitext-small-103-raw \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 2048 --update-freq 16 \
  --fp16 \
  --max-update 50000 \
  # --srcdict /home/gzc/fairseq-test/dict.txt;
