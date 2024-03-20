# Vanila Transformer

---
layout: single
title: "Vanila Transformer"
data: 2023-10-04 17:00:00
---

Transformer에 관한 잘 작성된 많은 글들이 있지만, 
본 Transformer 시리즈에서는 쉽게 설명하면서도 왜 이렇게 모델의 구조가 설계되었는 지 그리고 코드까지 같이 설명하여 이해할 수 있는 글이 되려고 합니다.

- Seq2Seq의 흐름
- Transformer - Overview
- Transformer - Deep Dive

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class SelfAttention(nn.Module):
    def __init__(self, input_dim: int) -> None:
        """Initialize the SelfAttention module.

        Args:
            input_dim (int): The size of the input dimension.

        The SelfAttention module consists of three linear layers that transform
        the input to create queries, keys, and values. This is a standard practice
        in attention mechanisms to capture different aspects of the input.
        """
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim

        # Linear transformations for creating queries, keys, and values
        self.query_transform = nn.Linear(input_dim, input_dim)
        self.key_transform = nn.Linear(input_dim, input_dim)
        self.value_transform = nn.Linear(input_dim, input_dim)

        # Softmax for normalizing attention scores to probabilities
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the SelfAttention module.

        Args:
            x (Tensor): The input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Tensor: The output tensor after applying self attention.

        The forward pass computes attention scores and applies these to
        the value vectors. The attention scores are scaled down by the square root
        of the input dimension to prevent large values that could push the softmax
        function into regions where it has extremely small gradients (saturation).
        """
        # Create query, key, and value vectors
        queries = self.query_transform(x)
        keys = self.key_transform(x)
        values = self.value_transform(x)

        # Calculate the attention scores with scaling for numerical stability
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        
        # Normalize the attention scores to probabilities
        attention_probs = self.softmax(scores)
        
        # Apply the attention to the values
        weighted_values = torch.bmm(attention_probs, values)
        
        return weighted_values
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Implements a Multi-Head Attention mechanism.
    
    Args:
        input_dim (int): Size of the input dimension.
        num_heads (int): Number of attention heads.
        
    Attributes:
        num_heads (int): The number of separate attention heads.
        head_dim (int): Dimension of each attention head.
        query_layer (nn.Linear): Linear layer for projecting input to query space.
        key_layer (nn.Linear): Linear layer for projecting input to key space.
        value_layer (nn.Linear): Linear layer for projecting input to value space.
        output_layer (nn.Linear): Linear layer to project concatenated outputs.
    """
    
    def __init__(self, input_dim: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        # Combined linear layers for queries, keys, and values
        self.query_layer = nn.Linear(input_dim, input_dim)
        self.key_layer = nn.Linear(input_dim, input_dim)
        self.value_layer = nn.Linear(input_dim, input_dim)
        
        # Linear layer to combine the outputs of the different heads
        self.output_layer = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Multi-Head Attention layer.
        
        Args:
            x (torch.Tensor): A tensor of shape (batch_size, seq_length, input_dim)
                containing the input feature set.
                
        Returns:
            torch.Tensor: The output of the Multi-Head Attention mechanism, 
                a tensor of the same shape as the input x.
        """
        batch_size = x.size(0)
        
        # Apply the linear layers and split into multiple heads
        # Reshape from (batch_size, seq_length, input_dim) to
        # (batch_size, num_heads, seq_length, head_dim)
        queries = self.query_layer(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key_layer(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value_layer(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # The scaling factor (1/sqrt(head_dim)) helps in stabilizing gradients during training.
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)
        
        # Combine the attended values for each head
        head_output = torch.matmul(attention, values).transpose(1, 2).contiguous()
        
        # Concatenate the heads' outputs and reshape
        # The final shape is (batch_size, seq_length, input_dim)
        concatenated = head_output.view(batch_size, -1, self.num_heads * self.head_dim)
        
        # Final linear layer to combine heads' outputs
        output = self.output_layer(concatenated)
        return output
```

# 1. Overview 🤖

![Figure. Transformer 구조](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/Untitled.png)

Figure. Transformer 구조

**[Attention Is All You Need (Vaswani et al,. 2017)](https://arxiv.org/abs/1706.03762)**에 처음 소개된 이후 language translation, language modelling, and text classification 등 다양한 영역에 적용되고 있으며, 최근 영상 처리 영역에서도 또한 SOTA모델(예시: [ViT](https://arxiv.org/abs/2010.11929))로써 사용된다. CNN이나 RNN 구조 없이, **multi-head self-attention** 구조로 처리하는 게 가장 큰 특징이다. 이를 통해서 RNN과 달리 긴 문장을 효율적으로 병렬 처리할 수 있게 되었다.

> *In this work we propose the **Transformer**, a model architecture eschewing recurrence and instead **relying entirely on an attention mechanism to draw global dependencies between input and output.** The Transformer allows for significantly more **parallelization** and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

From **[Attention Is All You Need (Vaswani et al,. 2017)](https://arxiv.org/abs/1706.03762)***
> 

## 1.1. Previous Works

![Figure: 딥러닝 기반의 기계 번역 발전 과정](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/Screenshot_2023-02-24_at_4.58.13_PM.png)

Figure: 딥러닝 기반의 기계 번역 발전 과정

[Reference: [https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/seq2seq_4.mp4)

Reference: [https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

일반적으로, 기계 번역을 위한 딥러닝 모델들은 위와 같이 **encoder-decoder** 형태를 따르게 된다. 기존에는 RNN 구조를 활용하여 기계 번역을 위한 encoder-decoder 구조를 구현하였는데, RNN 구조에 따른 아래 두 문제가 존재하여 점차 attention mechanism으로 대체된다. 아래 기존 연구들을 통해서 그 문제점들과 발전 과정을 확인해본다. 

## 1.1.1 Sequence to Sequence Learning with Neural Networks (NIPS 2014)

![Screenshot 2023-02-24 at 4.55.16 PM.png](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/Screenshot_2023-02-24_at_4.55.16_PM.png)

- **Seq2Seq:** Multi-layered LSTM 기반의 encoder→**fixed-length context vector**→decoder 기계 번역 아키텍처
- **병렬화(parallelization)**에 제한적
    - Critical at **longer sequence lengths**, as **memory constraints** limit batching across examples
    - High latency, low throughput
- **Fixed-length context vector**: **고정된 크기**의 context vector 에 따른 **병목(bottleneck)** 현상
    - 하나의 **고정된 크기의** 문맥 벡터가 소스 문장의 모든 정보를 압축해서 가지고 있어야 하므로, 모델의 성능을 제한하는 bottleneck으로 작용하여 long-term dependency가 떨어짐. → 추후 attention network의 등장하게 된 배경

즉, 우리가 원하는 것은 **소스 문장의 모든 정보 + 현재 인코딩 단계까지 출력된 정보**를 입력값으로 넣고자 하는 것이다.  **

## 1.1.2 Neural Machine Translation by Jointly Learning to Align and Translate (ICLR 2015)

![Encodes the input sentence into a sequence of vectors and chooses a subset of these vectors adaptively while decoding the translation.](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/seq2seq.gif)

Encodes the input sentence into a sequence of vectors and chooses a subset of these vectors adaptively while decoding the translation.

<aside>
💡 **Attention mechanism** became an integral part of sequence modeling and transduction models in various tasks.

</aside>

Seq2Seq 모델에서 **fixed-length context vector**로 ****모든 필요한 정보를 압축하게 되면서 긴 문장을 처리하는 데 어려움을 겪는다. Seq2Seq + Attention 에서는 디코더가 인코더의 모든 스텝의 hidden states를받아 Attention 구조를 활용하여 decoding 과정을 수행한다.

- Encoder: a bidirectional RNN
- Decoder: proposed **attention model**
    - Attention mechanism: **a weighted sum of the input hidden states**

![Figure. Attention](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/1_wBHsGZ-BdmTKS7b-BtkqFQ.gif)

Figure. Attention

Encoder의 모든 스텝의 hidden states를 decoder의 hidden state와 score를 계산하여 문장에서 집중할 부분을 선택적으로 보고 prediction을 할 수 있게 한다.

⚠️ **Limitation**: RNN에 attention mechanism을 이용함에도 불구하고, 순차적으로 계산하는 RNN으로 인해 (1) 느린 학습 및 추론 속도와 (2) 충분하지 않은 성능이 제한적인 연구였다. 특히, 긴 문장을 처리하는 데 어려움을 갖고 있다.

## 1.2. Model Architecture

![Figure. The full model architecture of the transformer. (Image source: Fig 1 & 2 in [Vaswani, et al., 2017](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf).)](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/Untitled%201.png)

Figure. The full model architecture of the transformer. (Image source: Fig 1 & 2 in [Vaswani, et al., 2017](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf).)

### Input Embedding

- Input Embedding in Code
    
    ```python
    class Embeddings(nn.Module):
    	"""Convert the input tokens to vectors of dimension d_model."""
    	def __init__(self, d_model, vocab):
            super(Embeddings, self).__init__()
            self.embedding = nn.Embedding(vocab, d_model)
            self.d_model = d_model
    
    	def forward(self, x):
    		return self.embedding(x) * math.sqrt(self.d_model)
    ```
    

![[https://www.blossominkyung.com/deeplearning/transfomer-positional-encoding#c8143ffb-4ce2-4173-8f7d-501d3df723cc](https://www.blossominkyung.com/deeplearning/transfomer-positional-encoding#c8143ffb-4ce2-4173-8f7d-501d3df723cc)](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/Untitled%202.png)

[https://www.blossominkyung.com/deeplearning/transfomer-positional-encoding#c8143ffb-4ce2-4173-8f7d-501d3df723cc](https://www.blossominkyung.com/deeplearning/transfomer-positional-encoding#c8143ffb-4ce2-4173-8f7d-501d3df723cc)

Word Embedding: 각 단어를 index에 매칭하고 embedding layer를 거쳐 $d_{model} = 512$ 차원의 벡터로 변환한다. 일반적으로 pre-trained embedding layer를 활용한다.

![Word Embedding의 예시. 의미론적으로 비슷한 단어끼리 embedding space에서 cluster를 이루게 된다. [https://www.ruder.io/word-embeddings-1/](https://www.ruder.io/word-embeddings-1/)](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/Untitled%203.png)

Word Embedding의 예시. 의미론적으로 비슷한 단어끼리 embedding space에서 cluster를 이루게 된다. [https://www.ruder.io/word-embeddings-1/](https://www.ruder.io/word-embeddings-1/)

### Positional Embedding

![[https://www.blossominkyung.com/deeplearning/transfomer-positional-encoding](https://www.blossominkyung.com/deeplearning/transfomer-positional-encoding)](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/positional_encoding.gif)

[https://www.blossominkyung.com/deeplearning/transfomer-positional-encoding](https://www.blossominkyung.com/deeplearning/transfomer-positional-encoding)

Transformer는 RNN과 달리 입력 데이터를 순차적으로 처리하는 대신 병렬로 한번에 처리하기 때문에,  각 단어의 순서를 Transformer에 전달할 수단이 필요하다. Positional embedding은 input embedding vector에 위치 정보를 담은 벡터$(\vec{p_t})$를 더해준다. 

![Untitled](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/Untitled%204.png)

![Reference: [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/Untitled%205.png)

Reference: [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)

- 부록: Positional Embedding vs. Learned Positional Embedding
    
    Transformer는 sinusoid-wave 기반의 positional encoding을 활용한다. 이는 positional encoding layer를 학습하는 것과 비교하면 다음과 같았다.
    
    1. positional encoding layer 를 학습하는 것과 성능 상의 유의미한 차이가 없었다.
    2. sinusoid-wave 기반의 방법은 학습할 때 보지 못했던 긴 문장을 처리하는 데 더 유리할 것으로 예상된다.
    
    참고. [What DoPosition Embeddings Learn? AnEmpirical Study of Pre-Trained Language Model Positional Encoding](https://arxiv.org/abs/2010.04903)
    
- Positional Embedding in Code
    
    ```python
    class PositionalEncoding(nn.Module):
        def __init__(self, dim_model, dropout_p, max_len):
            super().__init__()
            # 드롭 아웃
            self.dropout = nn.Dropout(dropout_p)
    
            # Encoding - From formula
            pos_encoding = torch.zeros(max_len, dim_model)
            positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
            division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
    
            pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
            pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
    
            # Saving buffer (same as parameter without gradients needed)
            pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
            self.register_buffer("pos_encoding", pos_encoding)
    
        def forward(self, token_embedding: torch.tensor) -> torch.tensor:
            # Residual connection + pos encoding
            return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
    ```
    

### TransformerEncoder

![Untitled](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/Untitled%206.png)

N = 6의 동일한 layer로 구성되며, 각 layer는 2개의 sub-layer (multi-head self-attention and position-wise feedforward networks)로 구성되어 있다. 각 두 가지 sub-layer는 다음과 같다.

- **Sub-layer (1): multi-head self-attention mechanism**
    
    ![[https://github.com/jessevig/bertviz](https://github.com/jessevig/bertviz)](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/attention.gif)
    
    [https://github.com/jessevig/bertviz](https://github.com/jessevig/bertviz)
    
    Attention 방식을 통해 문장 내에서 어떤 부분에 얼만큼 attention을 가져야 하는 지 학습하게 된다. 
    
    ![Screenshot 2023-02-24 at 3.13.43 PM.png](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/Screenshot_2023-02-24_at_3.13.43_PM.png)
    
    $$
    Attention(Q,K,V) = Softmax(\frac{QK^T}{\sqrt{d_k}})V
    $$
    
    > Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.
    > 
    
    ### 왜 “Multi-head” 일까?
    
    ![multi-head.gif](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/multi-head.gif)
    
    ![As we encode the word "it", one attention head is focusing most on "the animal", while another is focusing on "tired" -- in a sense, the model's representation of the word "it" bakes in some of the representation of both "animal" and "tired".](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/Untitled%207.png)
    
    As we encode the word "it", one attention head is focusing most on "the animal", while another is focusing on "tired" -- in a sense, the model's representation of the word "it" bakes in some of the representation of both "animal" and "tired".
    
    문장은 모호한 경우가 많기 때문에, 다양한 관점(multi-head)에서 문장을 이해해야한다.. 위 그림에서도, 첫 번째 self-attention은 `it`과 `animal` 의 attention 값이 높았지만, 두 번째 self-attention은 `it`과 `tired`가 attention이 높았다.
    
    - Self-attention illustrations
        
        ![[https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/1__92bnsMJy8Bl539G4v93yg.gif)
        
        [https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)
        
    - **Muti-head attention details**
        
        ![Untitled](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/Untitled%208.png)
        
        $$
        Attention(Q,K,V) = Softmax(\frac{QK^T}{\sqrt{d_k}})V
        $$
        
- **Sub-layer (2) position-wise fully connected feed-forward network**
    
    $$
    FFN(x)=max(0, xW_1+b_1)W_2+b_2
    $$
    
    ![Pointwise-ff-nn.gif](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/Pointwise-ff-nn.gif)
    
    - Code
        
        ```python
        class PointwiseFeedForward(nn.Module):
            def __init__(self,d_model, d_linear,dropout = 0.2):
                super(PointwiseFeedForward,self).__init__()
                self.W1 = nn.Linear(d_model,d_linear)
                self.W2 = nn.Linear(d_linear, d_model)
                self.dropout = nn.Dropout(dropout)
                
            def forward(x):
                relu = F.relu(self.W1(x))
                output = self.dropout(self.W2(x))
                return output
        ```
        

또한, 각 sub-layer은 뒤이어 Add & Norm (=LayerNorm(x + Sublayer(x)))을 따른다. 

- Details on LayerNormalization
    
    For $X=[x_1,x_2,...,x_m]$, LayerNorm normalizes each $x_i$ across all its features such that each sample $x_i$ has 0 mean and unit variance.
    
    ```python
    
    ```
    

### TransformerDecoder

![transformer_decoding_1.gif](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/transformer_decoding_1.gif)

![transformer_decoding_2.gif](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/transformer_decoding_2.gif)

![Untitled](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/Untitled%209.png)

N = 6의 동일한 layer로 구성되며, 각 layer는 3개의 sub-layer (`multi-head self-attention`, `multi-head cross-attention`, and `position-wise feedforward networks`)로 구성되어 있다. 각 세 가지 sub-layer는 다음과 같다.

![Untitled](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/Untitled%2010.png)

- **Sub-layer (1): masked multi-head self-attention**
    
    ![Untitled](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/Untitled%2011.png)
    
    ![Untitled](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/Untitled%2012.png)
    
    Transformer는 전체 입력값을 전달받기 때문에 과거 시점의 입력값을 예측할 때 미래 시점의 입력값까지 참고할 수 있다는 문제가 있습니다. 이를 방지하기 위해 `Look-ahead Mask` 기법을 이용한 Masked Attention 을 활용합니다.
    
    Decoder는 “**Masked**” multi-head attention을 활용해 i 번째 토큰은 i번째 이후 값에는 independent할 수 있도록 한다. 구체적으로, attention score 행렬의 대각선 윗부분을 `-inf` 로 두고 (Look-ahead Mask) Softmax를 취하여 해당 요소들의 Attention Weight들을 0으로 만들어 Attention Value를 계산할 때 미래 시점의 값을 고려하지 않도록 만들어준다.
    
- **Sub-layer (2): multi-head cross-attention (Encoder-Decoder attention)**
    - **`self`**-attention이 아닌 **`cross`**-attention 을 사용한다.
    - 인코더의 마지막 출력 값이 모든 디코더에 입력 값 중 하나로 사용.
        
        
- **Sub-layer (3) position-wise fully connected feed-forward network**

## 1.3. Pros and Cons

<aside>
<img src="https://super.so/icon/dark/align-left.svg" alt="https://super.so/icon/dark/align-left.svg" width="40px" /> `Summary` Transformer는 long-sequence modeling 에 유리하나, computational cost, memory requirement, 그리고 많은 학습 데이터 요구사항을 고려한 모델링이 필요함.

</aside>

<aside>
👍🏻 Pros

1. Long-sequence modeling
2. 병렬 처리로 빠른 학습
3. 적은 inductive bias로 인한 일반화된 학습에 유리
</aside>

<aside>
👎🏻  Cons

1. Computational cost
2. Memory-intensive
3. 많은 학습 데이터 요구
4. Auto-regressive 예측으로 인한 추론 시 오래 걸린다.
</aside>

## 1.4. Transformer 연구 동향

1. Model Efficiency
    - Self-Attention 모듈은 긴 입력 시퀀스에 대한 높은 연산과 메모리 비용을 요구
    - Sparse Attention 방식으로 해결하는 방향: [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)
2. Model Generalization
    - Inductive bias가 적지만 많은 학습 데이터가 있어야함.
    - Self-supervised learning을 통한 사전 학습을 통해 일반화 성능 극대화.
3. Model Adaptation
    - Specific한 downstream task에 적용하기 위한 연구

---

## 2.2. **Training Strategy**

### 2.2.1 Pre-trained Transformer

![Untitled](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/Untitled%2013.png)

<aside>
💡 “Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by *pre-training on a large corpus of text followed by fine-tuning on a specific task*.”
참조: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

</aside>

최근 BERT, GPT 같은 모델이 주목을 받게 된 이유는 성능 때문입니다. 이 모델들을 사용하면 문서 분류, 개체명 인식 등 어떤 태스크든지 점수가 이전 대비 큰 폭으로 오르기 때문인데요. BERT, GPT 따위의 부류는 **미리 학습된 언어 모델(pretrained language model)**이라는 공통점이 있습니다. [https://jalammar.github.io/illustrated-bert/](https://jalammar.github.io/illustrated-bert/)

일반적으로 Transformer는 많은 학습 데이터를 요구하며, 이는 상대적으로 가용 데이터가 적은 HMG-EV 프로젝트에는 Pre-trained Transformer를 필요로 합니다. 대표적인 pre-training 방식은 skip-gram, masked model 등이 있습니다.

![Masked model 예시](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/Untitled%2014.png)

Masked model 예시

**마스크 모델(Masked  Model)**은 학습 대상 문장에 빈칸을 만들어 놓고 해당 빈칸에 올 단어로 적절한 단어가 무엇일지 분류하는 과정으로 학습합니다. [BERT: Bidirectional Encoder Representations from Transformers](https://arxiv.org/abs/1810.04805)가 마스크 언어모델로 pre-train되는 대표적인 모델입니다. ****

![Skip-gram model 예시](Vanila%20Transformer%203844c831810e40a1b93ed48573f04fb7/Untitled%2015.png)

Skip-gram model 예시

**스킵-그램 모델(Skip-Gram Model)**은 어떤 단어 앞뒤에 특정 범위를 정해 두고 이 범위 내에 어떤 단어들이 올지 분류하는 과정에서 학습합니다. 다음 그림은 컨텍스트로 설정한 단어(파란색 네모칸) 앞뒤로 두 개씩 보는 상황을 나타낸 예시입니다. Skim-Gram 방식을 활용한 대표적인 연구로는 [Word2Vec](https://arxiv.org/abs/1301.3781) 이 있습니다.****

유사하게 Time-series 데이터에 masked model, skip-gram model 방식을 적용한 학습이 가능합니다.

---

- Pre-training large transformers on massive amounts of data has been shown to improve their performance on downstream tasks, reducing the need for task-specific data.
- Transformer는 많은 computational cost와 memory를 요구할 수 있어, HMG-EV 에서 가용 가능한 수준의 computation 수준과 memory 크기에 맞춰 모델 개발을 필요로 할 수 있다.
- Self-attention은 sequence length에 대하여 $O(n^2)$의 computational complexity를 요구하기 때문에, 적정 수준의 sequence length를 결정해야한다. (즉, 너무 긴 시퀀스 데이터를 다루는 것은 computational cost로 인해 어려울 수 있다.)
- Transformer는 원래 NLP를 위해 개발되었고 SOTA 모델로 사용되고 있다. Time-series 에서도 가장 적합한 모델일 지에 대해서는 추가적인 연구가 필요할 수 있다. (참조: [Are Transformers the most Effective model for Time Series Forecasting?](https://arxiv.org/abs/2205.13504))

**References**

- [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- [https://nlp.seas.harvard.edu/2018/04/03/attention.html](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- [https://huggingface.co/blog/time-series-transformers](https://huggingface.co/blog/time-series-transformers)
- [https://lilianweng.github.io/posts/2018-06-24-attention/](https://lilianweng.github.io/posts/2018-06-24-attention/)
- [https://github.com/bentrevett/pytorch-seq2seq/blob/master/6 - Attention is All You Need.ipynb](https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb)
- [https://tigris-data-science.tistory.com/entry/차근차근-이해하는-Transformer5-Positional-Encoding](https://tigris-data-science.tistory.com/entry/%EC%B0%A8%EA%B7%BC%EC%B0%A8%EA%B7%BC-%EC%9D%B4%ED%95%B4%ED%95%98%EB%8A%94-Transformer5-Positional-Encoding)
- [https://mohitkpandey.github.io/posts/2020/11/trfm-code/](https://mohitkpandey.github.io/posts/2020/11/trfm-code/)Table of Contents

---