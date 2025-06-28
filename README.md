# Transformer-in-Pytorch

-----------------
This repository contains a PyTorch implementation of the Transformer model, as introduced in the paper
"Attention is All You Need" (https://arxiv.org/abs/1706.03762). The Transformer model is widely used in NLP tasks and has become the foundation for several modern architectures like BERT, GPT, and T5.


### 1. Encoder

The Encoder is responsible for processing the input sequence and extracting the meaningful context.

- **Input Embedding**: 
  The input tokens are initially transformed into continuous vectors of fixed dimensionality, typically using an embedding layer. The embeddings are designed to map each token to a high-dimensional space (for example, 512-dimensional vectors).

- **Positional Encoding**:
  Since the Transformer does not process data sequentially (like RNNs or LSTMs), it has no inherent notion of the order of tokens in a sequence. To solve this, positional encodings are added to the input embeddings. These encodings are vectors that encode the position of tokens in the sequence. In the original Transformer, sinusoidal functions are used for positional encoding, but other forms can be used depending on the task.

  The positional encoding vectors are element-wise added to the token embeddings, allowing the model to differentiate between tokens at different positions.

- **Multi-Head Self-Attention**:
  This is the core mechanism of the Transformer. In self-attention, each token in the input sequence can attend to every other token. The attention mechanism computes a weighted sum of all tokens, where the weights are determined by how much focus a particular token should have on others.

  The attention score for each token pair is computed using three vectors: **Query (Q)**, **Key (K)**, and **Value (V)**. The Transformer employs multi-head attention, which means multiple attention operations are performed in parallel, allowing the model to focus on different parts of the sequence at the same time. The final output is the concatenation of the attention outputs from each head, followed by a linear transformation.

  The attention mechanism operates as follows:
  - Compute the **scaled dot-product** of the query and key vectors:  
    \[
    Attention(Q, K, V) = softmax \left( \frac{QK^T}{\sqrt{d_k}} \right) V
    \]
    where \(d_k\) is the dimensionality of the key vectors. This ensures that the attention scores are normalized.
  - The attention scores are used to create a weighted sum of the values, which is then passed to the next layer.

- **Feedforward Neural Network**:
  After the multi-head attention, each token's representation is passed through a fully connected feedforward neural network. This network consists of two layers with a ReLU activation function in between:
  \[
  \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
  \]
  The feedforward network helps the model learn non-linear representations of the input.

- **Residual Connection & Layer Normalization**:
  Each sub-layer (multi-head attention and feedforward network) has a residual connection around it, and layer normalization is applied to the sum of the input and output of each sub-layer. This helps stabilize training and improves convergence.

  The sub-layers are applied sequentially:
  1. Input -> Multi-Head Attention -> Residual Connection -> Layer Normalization
  2. Output -> Feedforward Neural Network -> Residual Connection -> Layer Normalization

### 2. Decoder

The Decoder generates the output sequence from the representations produced by the encoder. It is similar to the encoder, but it has additional mechanisms to attend to the encoder's output.

- **Masked Multi-Head Self-Attention**:
  Similar to the encoder's attention mechanism, the decoder has a self-attention layer. However, in this case, it is *masked* to ensure that tokens in the decoder only attend to previous tokens in the output sequence. This is necessary during training when the model is predicting each token one by one and prevents peeking into future tokens.

- **Encoder-Decoder Attention**:
  The decoder contains an additional attention mechanism that attends to the encoder's output. This is where the decoder learns which parts of the input sequence are relevant to generate the current output token. This is computed by performing attention between the decoder's queries (from previous decoder layers) and the encoder's keys and values.

- **Feedforward Neural Network**:
  Just like the encoder, the decoder includes a position-wise feedforward network after the attention layers.

- **Output Layer**:
  The final layer of the decoder is a linear layer followed by a softmax activation function. This produces the predicted probability distribution for each token in the vocabulary, given the decoder's internal state.

### 3. Scaled Dot-Product Attention

The attention mechanism used in the Transformer is the **Scaled Dot-Product Attention**. This is a core component of both the encoder and decoder, and it operates on three components: the query (Q), the key (K), and the value (V). These components are derived from the input sequence through learned linear transformations.

The attention score between a query and a key is computed as the dot product of these vectors, scaled by the square root of their dimensionality (\( \sqrt{d_k} \)) to prevent the values from growing too large, which could result in gradients that are too small for effective learning.

The output of the attention layer is the weighted sum of the values, where the weights are determined by the softmax of the attention scores.

### 4. Multi-Head Attention

Multi-Head Attention allows the model to jointly attend to information from different representation subspaces at different positions. Instead of having a single set of attention weights, the model learns multiple sets (called heads) in parallel.

- For each head:
  - The queries, keys, and values are projected into lower-dimensional spaces.
  - Attention is calculated in parallel for each set of queries, keys, and values.
  
- The outputs of all attention heads are concatenated and linearly transformed to produce the final result.

The intuition behind multi-head attention is that it allows the model to focus on different parts of the input sequence simultaneously, capturing a wider range of dependencies.

### 5. Final Layer Normalization and Linear Projection

After the multi-head attention and feedforward operations in both the encoder and decoder, the final outputs undergo layer normalization. This normalization ensures that the activations remain within a stable range, which helps to avoid issues like vanishing/exploding gradients.

Finally, in the decoder, the output is projected back into the vocabulary space through a linear layer, followed by a softmax to obtain probabilities for each token in the vocabulary.


Usage
-----
To train the Transformer model on your own dataset, follow these steps:

1. Prepare your dataset in the format expected by the `data.py` script. You can define your dataset and DataLoader for training and validation.

2. Define your hyperparameters in the `config.py` file.

3. Train the model:

   .. code-block:: bash
      python train.py --epochs 10 --batch_size 32

4. Evaluate the model on a test set:

   .. code-block:: bash
      python evaluate.py --model_path saved_model.pth





Training
--------
### Hyperparameters

You can customize the model's hyperparameters through the `config.py` file, including:

- `d_model`: The dimensionality of the model (default: 512)
- `n_heads`: The number of attention heads in each attention layer (default: 8)
- `num_layers`: The number of layers in both the encoder and decoder (default: 6)
- `dropout_rate`: Dropout probability to prevent overfitting (default: 0.1)
- `lr`: Learning rate (default: 1e-4)


Evaluation
----------
Once the model is trained, you can evaluate its performance on a held-out test set using the `evaluate.py` script.

### Example:

.. code-block:: bash
   python evaluate.py --model_path saved_model.pth --test_file test_data.txt

Results
-------
- This implementation has been tested on machine translation tasks, with results comparable to those reported in the original paper.
- Future work includes implementing fine-tuning strategies for specific NLP tasks.

License
-------
This project is licensed under the MIT License - see the `LICENSE` file for details.

References
----------
https://arxiv.org/abs/1706.03762
https://www.youtube.com/watch?v=ISNdQcPhsts&t=10575s
https://github.com/hkproj/pytorch-transformer
