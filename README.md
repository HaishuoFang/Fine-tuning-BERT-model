# FINE-TUNING BERT MODEL 

This reposity is a guidance for finetuning bert model.

- modify `modeling.py` to adjust to the need of different tasks.Mainly,I added another pooler layer for sequence output.
- use `tf.estimator.Estimator` instead of `tf.contrib.tpu.TPUEstimator`.
- overwrite `DataProcessor` for different tasks.

## Classification

For classificaiton task, we use `[CLS]` as the aggregate representation of the sentence and feed it into a clssification 
layer(softmax layer) where the only new parameters from during fine-tuning.
`run_classifier.py` is the main script to use for classification task. In this script,I implement two kinds of data-process, 
binary classificaiton and multiclass classificaition.

> The first thing you need to do is to download BERT model from [BERT](https://github.com/google-research/bert) and then
put it under the model_dir directory.

The example is:
```
python run_classifier.py\
    --task_name=binary \
    --do_train=true \
    --data_dir=data \
    --vocab_file=model_dir/vocab.txt \
    --bert_config_file=model_dir/bert_config.json \
    --init_checkpoint=model_dir/bert_model.ckpt \
    --max_seq_length=25 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=/tmp/binary_output/
```

## Sequence labeling

For sequence labeling task, we feed every token's hidden representation into a classification layer over the label 
set without a CRF layer.You can try to add a CRF layer before the classificaiton layer.
The script is `run_ner.py`.

The example is similar to the example above.The parameter needing to be changed is `task_name` and `output_dir`.

```
python run_ner.py\
    --task_name=ner \
    --do_train=true \
    --data_dir=data \
    --vocab_file=model_dir/vocab.txt \
    --bert_config_file=model_dir/bert_config.json \
    --init_checkpoint=model_dir/bert_model.ckpt \
    --max_seq_length=25 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=/tmp/ner_output/
``` 
Tips: you can merge these two scripts to one.you should pay attenion to loss function and code structure.

I have got  better performance in my tasks by fine-tuning the BERT model.So it is worth trying!