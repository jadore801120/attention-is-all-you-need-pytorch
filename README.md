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

**BPE related parts are not yet fully tested.**


If there is any suggestion or error, feel free to fire an issue to let me know. :)


# Requirement
- python 3.4+
- pytorch 1.3.1
- torchtext 0.4.0
- spacy 2.2.2+
- tqdm
- dill
- numpy


# Usage

## WMT'16 Multimodal Translation: de-en

An example of training for the WMT'16 Multimodal Translation task (http://www.statmt.org/wmt16/multimodal-task.html).

### 0) Download the spacy language model.
```bash
# conda install -c conda-forge spacy 
python -m spacy download en
python -m spacy download de
```

### 1) Preprocess the data with torchtext and spacy.
```bash
python preprocess.py -lang_src de -lang_trg en -share_vocab -save_data m30k_deen_shr.pkl
```

### 2) Train the model
```bash
python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 256 -warmup 128000 -epoch 400
```

### 3) Test the model
```bash
python translate.py -data_pkl m30k_deen_shr.pkl -model trained.chkpt -output prediction.txt
```

## [(WIP)] WMT'17 Multimodal Translation: de-en w/ BPE 
### 1) Download and preprocess the data with bpe:

> Since the interfaces is not unified, you need to switch the main function call from `main_wo_bpe` to `main`.

```bash
python preprocess.py -raw_dir /tmp/raw_deen -data_dir ./bpe_deen -save_data bpe_vocab.pkl -codes codes.txt -prefix deen
```

### 2) Train the model
```bash
python train.py -data_pkl ./bpe_deen/bpe_vocab.pkl -train_path ./bpe_deen/deen-train -val_path ./bpe_deen/deen-val -log deen_bpe -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 256 -warmup 128000 -epoch 400
```

### 3) Test the model (not ready)
- TODO:
	- Load vocabulary.
	- Perform decoding after the translation.
---
# Performance
## Training

<p align="center">
<img src="https://imgur.com/rKeP1bb.png" width="400">
<img src="https://imgur.com/9je3X6U.png" width="400">
</p>

- Parameter settings:
  - default parameter and optimizer settings
  - label smoothing 
  - target embedding / pre-softmax linear layer weight sharing. 

- Elapse per epoch (on NVIDIA Titan X):
  - Training set: 0.888 minutes
  - Validation set: 0.011 minutes
  
## Testing 
- coming soon.
---
# TODO
  - Evaluation on the generated text.
  - Attention weight plot.
---
# Acknowledgement
- The byte pair encoding parts are borrowed from [subword-nmt](https://github.com/rsennrich/subword-nmt/).
- The project structure, some scripts and the dataset preprocessing steps are heavily borrowed from [OpenNMT/OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).
- Thanks for the suggestions from @srush, @iamalbert, @Zessay, @JulesGM and @ZiJianZhao.
