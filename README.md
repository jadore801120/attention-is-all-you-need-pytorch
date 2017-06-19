# Attention is all you need: A Pytorch Implementation

This is a PyTorch implementation of the Transformer model in "[Attention is All You Need](https://arxiv.org/abs/1706.03762)". 

State-of-the-art performance on **WMT 2014 English-to-German translation task**. (2017/06/12)

A novel sequence to sequence framework utilizes the *self-attention mechanism*, instead of Convolution operation or Recurrent structure.
> To learn more about self-attention mechanism, you could read "[A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)".

<img src="http://imgur.com/1krF2R6.png" width="250">

The project is still in WIP, now only support training.

Translating (Beam search) will be available soon.

# Usage

## 0) Prepare the data
```bash
python preprocess.py -train_src train.src.txt -train_tgt train.tgt.txt -valid_src valid.src.txt -valid_tgt valid.tgt.txt -output output.pt
```

## 1) Training
```bash
python train.py -data output.pt -embs_share_weight -proj_share_weight
```
## 2) Testing
### TODO
  - **Beam search** 

# Requirement
- python 3.4+
- pytorch 0.1.12
- tqdm
- numpy

# Acknowledgement
- The project structure is heavily learned from [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)

---
If there is any suggestion or error, feel free to fire an issue to let me know. :)
