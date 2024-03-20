# Vanila Transformer

---
layout: single
title: "Vanila Transformer"
data: 2023-10-04 17:00:00
---

# Positional Embedding

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
