
make vocab file for training. train.all means a merged file from train.req & train.rep

``` bash
  	python generate_vocab.py < data\train.req > data\vocab.req
  	python generate_vocab.py < data\train.rep > data\vocab.rep
```

``` bash
	python -m nmt.nmt --attention=scaled_luong --src=req --tgt=rep --vocab_prefix=data/vocab --train_prefix=data/train --dev_prefix=data/dev --test_prefix=data/test --out_dir=nmt_model --num_train_steps=12000 --steps_per_stats=100 --num_layers=4 --num_units=128 --dropout=0.2 --metrics=bleu
```
