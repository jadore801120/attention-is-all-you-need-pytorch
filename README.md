# Attention is all you need: A Pytorch Implementation

This is a PyTorch implementation of the Transformer model in "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017). 


A novel sequence to sequence framework utilizes the **self-attention mechanism**, instead of Convolution operation or Recurrent structure, and achieve the state-of-the-art performance on **WMT 2014 English-to-German translation task**. (2017/06/12)

> The official Tensorflow Implementation can be found in: [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py).

> To learn more about self-attention mechanism, you could read "[A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)".

<p align="center">
<img src="http://imgur.com/1krF2R6.png" width="250">
</p>


The project support training and translation with trained model now.

Note that this project is still a work in progress.


If there is any suggestion or error, feel free to fire an issue to let me know. :)


# Requirement
- python 3.4+
- pytorch 0.2.0
- tqdm
- numpy


# Usage

## Some useful tools:

The example below uses the Moses tokenizer (http://www.statmt.org/moses/) to prepare the data and the moses BLEU script for evaluation.

```bash
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.de
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en
sed -i "s/$RealBin\/..\/share\/nonbreaking_prefixes//" tokenizer.perl
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl
```

## WMT'16 Multimodal Translation: Multi30k (de-en)

An example of training for the WMT'16 Multimodal Translation task (http://www.statmt.org/wmt16/multimodal-task.html).

### 0) Download the data.

```bash
mkdir -p data/multi30k
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz &&  tar -xf training.tar.gz -C data/multi30k && rm training.tar.gz
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz && tar -xf validation.tar.gz -C data/multi30k && rm validation.tar.gz
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz && tar -xf mmt16_task1_test.tar.gz -C data/multi30k && rm mmt16_task1_test.tar.gz
```

### 1) Preprocess the data.
```bash
for l in en de; do for f in data/multi30k/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi;  done; done
for l in en de; do for f in data/multi30k/*.$l; do perl tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok; done; done
python preprocess.py -train_src data/multi30k/train.en.atok -train_tgt data/multi30k/train.de.atok -valid_src data/multi30k/val.en.atok -valid_tgt data/multi30k/val.de.atok -save_data data/multi30k.atok.low.pt
```

### 2) Train the model
```bash
python train.py -data data/multi30k.atok.low.pt -save_model trained -save_mode best -proj_share_weight 
```
> If your source and target language share one common vocabulary, use the `-embs_share_weight` flag to enable the model to share source/target word embedding. 

### 3) Test the model
```bash
python translate.py -model trained.chkpt -vocab data/multi30k.atok.low.pt -src data/multi30k/test.en.atok
```
---
# Performance
## Training

<p align="center">
<img src="https://imgur.com/A6byNQ8.png" width="400">
<img src="https://imgur.com/5z4f4tw.png" width="400">
</p>

- Parameter settings:
  - batch_size=64
  - d_inner_hid=1024
  - d_k=64
  - d_v=64
  - d_model=512
  - d_word_vec=512
  - dropout=0.1
  - embs_share_weight=False
  - n_head=8
  - n_layers=6
  - n_warmup_steps=4000
  - proj_share_weight=True

- Elapse per epoch  (on NVIDIA Titan X):
  - Training set: 1.38 min
  - Validation set: 0.016 min
  
## Testing 
- coming soon.
---
# TODO
  - Label smoothing
  - Evaluation on the generated text.
  - Attention weight plot.
---
# Acknowledgement
- The project structure, some scripts and the dataset preprocessing steps are heavily borrowed from [OpenNMT/OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).
- Thanks for the suggestions from @srush, @iamalbert and @ZiJianZhao.
