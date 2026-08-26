
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import BertTokenizer, AutoTokenizer

import math
import numpy as np

"""
    Constructs positional encodings
    Positional Encodings inject some information about the relative or absolute position of the tokens in the sequence.

    Args:
      emb_size: Integer
        Specifies embedding size
      dropout: Float
        Specifies Dropout probability hyperparameter
      max_len: Integer
        Specifies maximum sequence length



Input
    ↓
Tokenization
    ↓
TransformerClassifier
    ↓
    ├── Token Embedding
    ├── Positional Encoding
    ├── TransformerBlock * depth
    │   ├── Multi-Head Self-Attention
    │   │   ├── SelfAttention
    │   │   │   ├── to_queries (Linear)
    │   │   │   ├── to_keys (Linear)
    │   │   │   ├── to_values (Linear)
    │   │   │   └── DotProductAttention
    │   │   │       ├── Score = Q·K^T / √d_k
    │   │   │       ├── Softmax
    │   │   │       ├── Dropout
    │   │   │       └── Output = Attention * V
    │   │   └── unify_heads (Linear)
    │   ├── Add & Norm (Residual Connection + LayerNorm)
    │   ├── Feed-Forward (MLP)
    │   │   ├── Linear (embed_dim → 2*embed_dim)
    │   │   ├── ReLU
    │   │   └── Linear (2*embed_dim → embed_dim)
    │   └── Add & Norm (Residual Connection + LayerNorm)
    ↓
Global Average Pooling (seq_length)
    ↓
Classification Head (Linear)
    ↓
Log-Softmax
    ↓
Output 
"""




class DotProductAttention(nn.Module):
    def __init__(self, dropout: float):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def score(self, queries:Tensor, keys:Tensor) -> Tensor:
        return torch.bmm(queries,keys.transpose(1,2))/math.sqrt(queries.shape[-1])
        # https://docs.pytorch.org/docs/2.13/generated/torch.bmm.html
        # If input is a ( b × n × m ) (b×n×m) tensor, mat2 is a ( b × m × p ) (b×m×p) tensor, out will be a ( b × n × p ) (b×n×p) tensor.
    def forward(self, queries:Tensor, keys:Tensor, values:Tensor, batch_size: int,
                num_heads: int, seq_length: int, embedding_size: int) -> Tensor:


        #before transpose  (batch_size, seq_length, num_heads, embedding_size)
        #after transpose (batch_size, num_heads, seq_length, embedding_size)

        # before: (2, 4, 3, 8) → batch=2, heads=4, seq=3, embed=8
        # then: (8, 3, 8) → 2*4=8, seq=3, embed=8
        keys = keys.transpose(1,2).contiguous().view(batch_size*num_heads, seq_length, embedding_size)
        queries = queries.transpose(1,2).contiguous().view(batch_size*num_heads, seq_length, embedding_size)
        values = values.transpose(1,2).contiguous().view(batch_size*num_heads, seq_length, embedding_size)
    
        soft_max = F.softmax(self.score(queries, keys),dim=2)

        out = torch.bmm(self.dropout(soft_max), values).view(batch_size,num_heads, seq_length, embedding_size)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, num_heads * embedding_size)

        return out

class SelfAttention(nn.Module):
    def __init__(self, embedding_size, dropout : float = 0.1, num_heads: int = 8, ):
        super(SelfAttention, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        # for head in range(num_heads):
        #     to_key_head = nn.Linear(embedding_size, embedding_size)  # ۸ times!

        self.to_key = nn.Linear(embedding_size,embedding_size * num_heads,bias=False)
        self.to_querie = nn.Linear(embedding_size,embedding_size * num_heads,bias=False)
        self.to_value = nn.Linear(embedding_size,embedding_size * num_heads,bias=False)
        self.unify_heads = nn.Linear(embedding_size * num_heads, embedding_size)


        self.attention = DotProductAttention(dropout)

    def forward(self, x : Tensor)-> Tensor:

        batch_size, seq_length, embedding_size = x.size()
        num_heads = self.num_heads
        # INPUT: (batch_size, seq_length, embedding_size) (2, 3, 128)
        # (batch, seq, embedding_size * num_heads) (2, 3, 1024)
        # (batch, seq, num_heads, embedding_size) (2, 3, 8, 128)
        keys = self.to_key(x).view(batch_size, seq_length,num_heads ,embedding_size)
        queries = self.to_querie(x).view(batch_size, seq_length,num_heads ,embedding_size)
        values = self.to_value(x).view(batch_size, seq_length,num_heads ,embedding_size)

        out = self.attention(queries, keys, values, batch_size, num_heads, seq_length, embedding_size)
        
        return self.unify_heads(out)


class PositionalEncoding(nn.Module):
  # Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
  """ Block initiating Positional Encodings """

  def __init__(self, emb_size, dropout=0.1, max_len=512):
    
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, emb_size)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-np.log(10000.0) / emb_size))

    # Each dimension of the positional encoding corresponds to a sinusoid.
    # The wavelengths form a geometric progression from 2π to 10000·2π.
    # This function is chosen as it's hypothesized that it would allow the model
    # to easily learn to attend by relative positions, since for any fixed offset k,
    # PEpos + k can be represented as a linear function of PEpos.
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe)

  def forward(self, x):
    """
    Defines network structure

    Args:
      x: Tensor
        Input sequence

    Returns:
      x: Tensor
        Output is of the same shape as input with dropout and positional encodings
    """
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)

class TransformerBlock(nn.Module):
    """Block to instantiate transformers."""

    def __init__(self, embed_dim: int, num_heads: int):
        """
        Initializes the Transformer block.

        Args:
            embed_dim (int): Size of attention embeddings.
            num_heads (int): Number of self-attention heads.
        """
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embed_dim, num_heads=num_heads)

        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)

        hidden_size = 2 * embed_dim  # This is a somewhat arbitrary choice
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass through the transformer block.

        Args:
            x (Tensor): Input tensor with shape (batch_size, seq_length, embed_dim).

        Returns:
            Tensor: Output tensor after processing through the transformer block.
        """
        attended = self.attention(x)
        x = self.norm_1(attended + x)  # First Add & Normalize layer

        feedforward = self.mlp(x)
        x = self.norm_2(feedforward + x)  # Second Add & Normalize layer

        return x


class Transformer(nn.Module):
    """Transformer Encoder network for classification."""

    def __init__(self, embed_dim: int, num_heads: int, depth: int, seq_length: int, num_tokens: int, num_classes: int):
        """
        Initializes the Transformer network.

        Args:
            embed_dim (int): Attention embedding size.
            num_heads (int): Number of self-attention heads.
            depth (int): Number of Transformer blocks.
            seq_length (int): Length of input sequence.
            num_tokens (int): Size of dictionary.
            num_classes (int): Number of output classes.
        """
        super(Transformer, self).__init__()

        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.token_embedding = nn.Embedding(num_tokens, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)

        transformer_blocks = [TransformerBlock(embed_dim=embed_dim, num_heads=num_heads) for _ in range(depth)]
        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.classification_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for classification within the Transformer network.

        Args:
            x (Tensor): Input tensor of tokenized words with shape (batch_size, seq_length).

        Returns:
            Tensor: Log-probabilities over classes with shape (batch_size, num_classes).
        """
        x = self.token_embedding(x) * np.sqrt(self.embed_dim)
        x = self.pos_enc(x)
        x = self.transformer_blocks(x)

        sequence_avg = x.mean(dim=1)
        x = self.classification_head(sequence_avg)
        logprobs = F.log_softmax(x, dim=1)
        return logprobs
class TransformerClassifier():
    def __init__(self, classes,  device=None,
                embed_dim = 128,
                num_heads = 8,
                depth = 8,
                max_len = 48,
                ):

        seq_length = max_len
        self.max_len = max_len

        
        self.device = device if device else torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Device of BertClassifier is: {self.device}")

        # try:
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        print(f"✔ tokenizer is loaded.")

        vocab_size = self.tokenizer.vocab_size
        self.model = Transformer(embed_dim, num_heads, depth, seq_length, vocab_size, len(classes)).to(self.device)



        print(f"✔ class Transformer model is loaded.")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        # except Exception as e:
        #     print(f"✘ error: {e}")


    def tokenize(self, input, max_length):
            
            
            return self.tokenizer(
                input,
                padding = 'max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )



