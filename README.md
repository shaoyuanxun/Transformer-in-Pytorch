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
  ```
  class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, device: torch.device):
        super().__init__()
        self.max_len = max_len
        # (max_len, d_model)
        self.pe = torch.zeros(self.max_len, d_model, requires_grad=False, device=device)
        # (max_len, 1)
        pos = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(dim=1)
        # (1, d_model/2)
        div_term = 1 / (10000 ** (torch.arange(0, d_model, step=2).float() / d_model))

        self.pe[:, 0::2] = torch.sin(pos * div_term)  # (max_len, d_model)
        self.pe[:, 1::2] = torch.cos(pos * div_term)  # (max_len, d_model)

    def forward(self, seq_len: int):
        assert seq_len <= self.max_len, "seq_len exceeds max_len"
        return self.pe[:seq_len, :]  # (seq_len, d_model)
  ```
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

```
class ScaleDocProductAttention(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = None
        if drop_prob is not None:
            self.dropout = nn.Dropout(drop_prob)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: [batch, n_head, q_len, d_model/n_head]
            k: [batch, n_head, kv_len, d_model/n_head]
            v: [batch, n_head, kv_len, d_model/n_head]
            mask: [batch, n_head, q_len, kv_len]

        Returns:
            v: [batch, n_head, q_len, d_model/n_head]
            score: [batch, n_head, q_len, kv_len]
        """
        assert (
            q.dim() == k.dim() == v.dim() == 4
        ), "input tensors must have 4 dimensions"
        assert k.size() == v.size(), "k and v must have the same shape"
        assert (
            q.size()[0:2] == k.size()[0:2] == v.size()[0:2]
        ), "q, k, and v must have the same batch size and number of heads"
        assert (
            q.size(3) == k.size(3) == v.size(3)
        ), "d_model/n_head must be the same for q, k, and v"

        dim_per_head = q.size()[3]

        # [batch, n_head, q_len, kv_len]
        scores = q @ k.transpose(2, 3) / math.sqrt(dim_per_head)

        if mask is not None:
            scores.masked_fill_(mask == 0, -1e6)
        scores = self.softmax(scores)  # [batch, n_head, q_len, kv_len]
        if self.dropout is not None:
            scores = self.dropout(scores)

        values = scores @ v  # [batch, n_head, q_len, d_model/n_head]

        return values, scores
```

```
class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_model: int, n_head: int, device: torch.device, mask=None, drop_prob=None
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dim_per_head = d_model // n_head
        assert d_model % n_head == 0, "d_model is not divided by n_head"
        self.mask = mask
        self.scores = None

        self.attention = ScaleDocProductAttention(drop_prob)

        self.q_linear = nn.Linear(d_model, d_model).to(device)
        self.k_linear = nn.Linear(d_model, d_model).to(device)
        self.v_linear = nn.Linear(d_model, d_model).to(device)
        self.out_linear = nn.Linear(d_model, d_model).to(device)

    def split(self, tensor):
        """
        Splits the input tensor into multiple heads.

        Args:
            tensor: [batch, seq_len, d_model]

        Returns:
            [batch, n_head, seq_len, d_model/n_head]
        """
        #     [batch, seq_len, d_model]
        #  -> [batch, seq_len, n_head, d_model/n_head]
        #  -> [batch, n_head, seq_len, d_model/n_head]
        return tensor.view(
            tensor.size(0), tensor.size(1), self.n_head, self.dim_per_head
        ).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: [batch, q_len, d_model]
            k: [batch, kv_len, d_model]
            v: [batch, kv_len, d_model]
            mask: defaults to None.

        Returns:
            out: [batch, q_len, d_model]
        """
        assert (
            q.dim() == k.dim() == v.dim() == 3
        ), "input tensors must have 3 dimensions"
        assert (
            q.size(0) == k.size(0) == v.size(0)
        ), "batch size must be the same for q, k, and v"
        assert (
            q.size(2) == k.size(2) == v.size(2)
        ), "d_model must be the same for q, k, and v"

        # [batch, q_len, d_model] -> [batch, q_len, d_model]
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        q = self.split(q)  # [batch, n_head, q_len, d_model/n_head]
        k = self.split(k)  # [batch, n_head, kv_len, d_model/n_head]
        v = self.split(v)  # [batch, n_head, kv_len, d_model/n_head]

        # val: (batch, n_head, q_len, d_model/n_head)
        # self.cores: (batch, n_head, q_len, kv_len)
        val, self.scores = self.attention(q, k, v, self.mask)
        batch = val.size(0)
        q_len = q.size(2)

        # (batch, n_head, q_len, d_model/n_head) -> (batch, q_len, d_model)
        val = val.transpose(1, 2).contiguous().view(batch, q_len, self.d_model)

        return val
```

- **Feedforward Neural Network**:
  After the multi-head attention, each token's representation is passed through a fully connected feedforward neural network. This network consists of two layers with a ReLU activation function in between:
  The feedforward network helps the model learn non-linear representations of the input.
<p align="center">
<img width="412" alt="image" src="https://github.com/user-attachments/assets/6e929384-4ebc-4247-9c06-a721527786e2" />
</p>

- **Residual Connection & Layer Normalization**:
  Each sub-layer (multi-head attention and feedforward network) has a residual connection around it, and layer normalization is applied to the sum of the input and output of each sub-layer. This helps stabilize training and improves convergence.
```
class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, drop_prob: float, device: torch.device):
        super().__init__()
        self.dropout = nn.Dropout(drop_prob)
        self.norm = LayerNorm(d_model, device)

    def forward(self, x, sublayer):
        # (B, seq_len, d_model)
        """
        Applies a residual connection followed by layer normalization and dropout.

        Args:
            x (torch.Tensor): Input tensor of shape (B, seq_len, d_model).
            sublayer (Callable): A sublayer function to apply to the normalized input.

        Returns:
            torch.Tensor: Output tensor of shape (B, seq_len, d_model).
        """

        return x + self.dropout(sublayer(self.norm(x)))
```
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

#### Training loss plot

![Screenshot from 2025-06-28 17-11-19](https://github.com/user-attachments/assets/e913b00e-331e-4245-b739-ec2a138b1b32)

#### Attention Score Viz

![Screenshot from 2025-06-28 17-29-01](https://github.com/user-attachments/assets/e7713bce-9269-4752-89af-3d81e2bc37a2)


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
