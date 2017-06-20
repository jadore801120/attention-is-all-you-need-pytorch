# Attention is all you need: A Pytorch Implementation

This is a PyTorch implementation of the Transformer model in "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017). 


A novel sequence to sequence framework utilizes the **self-attention mechanism**, instead of Convolution operation or Recurrent structure, and achieve the state-of-the-art performance on **WMT 2014 English-to-German translation task**. (2017/06/12)

> The official Tensorflow Implementation can be found in: [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py).

> To learn more about self-attention mechanism, you could read "[A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)".

<img src="http://imgur.com/1krF2R6.png" width="250">

The project support training and translation with trained model now.

Note that this project is still a work in progress.


If there is any suggestion or error, feel free to fire an issue to let me know. :)


# Requirement
- python 3.4+
- pytorch 0.1.12
- tqdm
- numpy


# Usage

## 0) Prepare the data
```bash
python preprocess.py -train_src train.src.txt -train_tgt train.tgt.txt -valid_src valid.src.txt -valid_tgt valid.tgt.txt -output data.pt
```

## 1) Training
```bash
python train.py -data data.pt -save trained.chkpt -save_mode best -embs_share_weight -proj_share_weight 
```

## 2) Testing
```bash
python translate.py -model trained.chkpt -vocab data.pt -src test.src.txt
```

---
### TODO
  - Evaluation on the generated text.
  - Attention weight plot.
---
# Acknowledgement
- The project structure and some scripts are heavily borrowed from [OpenNMT/OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
