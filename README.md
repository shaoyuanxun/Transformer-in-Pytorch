# Transformer-in-Pytorch

-----------------
This repository contains a PyTorch implementation of the Transformer model, as introduced in the paper
"Attention is All You Need" (https://arxiv.org/abs/1706.03762). The Transformer model is widely used in NLP tasks and has become the foundation for several modern architectures like BERT, GPT, and T5.

<p align="center">
<img width="681" alt="image" src="https://github.com/user-attachments/assets/23b0a5bb-171e-4f31-9321-46a331cc5af1" />
</p>

### 1. Encoder

The Encoder is responsible for processing the input sequence and extracting the meaningful context.

- **Input Embedding**: 
  The input tokens are initially transformed into continuous vectors of fixed dimensionality, typically using an embedding layer. The embeddings are designed to map each token to a high-dimensional space (for example, 512-dimensional vectors).

- **Positional Encoding**:
  Since the Transformer does not process data sequentially (like RNNs or LSTMs), it has no inherent notion of the order of tokens in a sequence. To solve this, positional encodings are added to the input embeddings. These encodings are vectors that encode the position of tokens in the sequence. In the original Transformer, sinusoidal functions are used for positional encoding, but other forms can be used depending on the task.

  The positional encoding vectors are element-wise added to the token embeddings, allowing the model to differentiate between tokens at different positions.
<p align="center">
<img width="456" alt="image" src="https://github.com/user-attachments/assets/da9a6a03-9dae-478b-88ca-587ddfcbd1e0" />
</p>

- **Multi-Head Self-Attention**:
  This is the core mechanism of the Transformer. In self-attention, each token in the input sequence can attend to every other token. The attention mechanism computes a weighted sum of all tokens, where the weights are determined by how much focus a particular token should have on others.

  The attention score for each token pair is computed using three vectors: **Query (Q)**, **Key (K)**, and **Value (V)**. The Transformer employs multi-head attention, which means multiple attention operations are performed in parallel, allowing the model to focus on different parts of the sequence at the same time. The final output is the concatenation of the attention outputs from each head, followed by a linear transformation.
  The attention mechanism operates as follows:
  - Compute the **scaled dot-product** of the query and key vectors:  
  - The attention scores are used to create a weighted sum of the values, which is then passed to the next layer.
<p align="center">
<img width="833" alt="image" src="https://github.com/user-attachments/assets/1625bef0-6264-44f8-94b0-befe6d9da402" />
<img width="460" alt="image" src="https://github.com/user-attachments/assets/8cea0573-eb3d-4c51-998a-d2f345666497" />
<img width="656" alt="image" src="https://github.com/user-attachments/assets/c40ddc53-485f-447a-a5a5-9a33d4fe92c9" />
</p>


- **Feedforward Neural Network**:
  After the multi-head attention, each token's representation is passed through a fully connected feedforward neural network. This network consists of two layers with a ReLU activation function in between:
  The feedforward network helps the model learn non-linear representations of the input.
<p align="center">
<img width="412" alt="image" src="https://github.com/user-attachments/assets/6e929384-4ebc-4247-9c06-a721527786e2" />
</p>

- **Residual Connection & Layer Normalization**:
  Each sub-layer (multi-head attention and feedforward network) has a residual connection around it, and layer normalization is applied to the sum of the input and output of each sub-layer. This helps stabilize training and improves convergence.

  The sub-layers are applied sequentially:
  1. Input -> Multi-Head Attention -> Residual Connection -> Layer Normalization
  2. Output -> Feedforward Neural Network -> Residual Connection -> Layer Normalization

### 2. Decoder

The Decoder generates the output sequence from the representations produced by the encoder. It is similar to the encoder, but it has additional mechanisms to attend to the encoder's output.

- **Masked Multi-Head Self-Attention**:
  Similar to the encoder's attention mechanism, the decoder has a self-attention layer. However, in this case, it is *masked* to ensure that tokens in the decoder only attend to previous tokens in the output sequence. This is necessary during training when the model is predicting each token one by one and prevents peeking into future tokens.

- **Encoder-Decoder Cross-Attention**:
  The decoder contains an additional attention mechanism that attends to the encoder's output. This is where the decoder learns which parts of the input sequence are relevant to generate the current output token. This is computed by performing attention between the decoder's queries (from previous decoder layers) and the encoder's keys and values.

- **Feedforward Neural Network**:
  Just like the encoder, the decoder includes a position-wise feedforward network after the attention layers.

- **Output Layer**:
  The final layer of the decoder is a linear layer followed by a softmax activation function. This produces the predicted probability distribution for each token in the vocabulary, given the decoder's internal state.

### 3. Final Layer Normalization and Linear Projection

After the multi-head attention and feedforward operations in both the encoder and decoder, the final outputs undergo layer normalization. This normalization ensures that the activations remain within a stable range, which helps to avoid issues like vanishing/exploding gradients.

Finally, in the decoder, the output is projected back into the vocabulary space through a linear layer, followed by a softmax to obtain probabilities for each token in the vocabulary.

### 3. Results


### 4. Usage
To train the Transformer model, follow these steps:

1. Define training parameters
```
      src/config/transformer_config.py
```
2. Train the model

```
      python run_transformer_train.py
```
3. Inference
Currently, supports greedy search and beam search.
```
      src/inference/*
```
4. Evaluation code

```
      src/evaluation/transformer_eval.py
```
5. Attention Scores Viz
```
      attentioin_viz.ipynb
```






References
----------
https://arxiv.org/abs/1706.03762
https://www.youtube.com/watch?v=ISNdQcPhsts&t=10575s
https://github.com/hkproj/pytorch-transformer
