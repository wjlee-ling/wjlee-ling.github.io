---
title: torch RNN/LSTM/GRU의 output과 hidden
search: true
description: 
date: 2022-10-17
categories:
    - TIL
tags: 
    - torch
---


`torch.nn.RNN` `torch.nn.LSTM` `torch.nn.GRU`은 `forward`의 결과물로 두 벡터를 돌려준다. pytorch의 공식 doc에 의하면 `output`과 `h_n`인데, 만드는 모델, 과제에 따라 사용해야 하는 벡터가 다르다. 지금까지 주로 huggingface 라이브러리를 썼기 때문에 torch.nn의 RNN/LSTM/GRU를 쓸 일은 별로 없었지만 아직도 양뱡향 LSTM 모델등은 일부 사용하는 경우가 있기에 이번 기회에 항상 헷갈렸던 `output`과 `'h_n`의 차이점에 대해 알아봤다.

> Note: `nn.LSTM` 은 `output`과 hidden_state, cell state가 담긴 튜플 `(h_n, c_n)`를 리턴한다.

아래 설명은 input이 `torch.nn.utils.rnn.PackedSequence` 객체로 되어 있다고 가정한다. 편이상 RNN/LSTM/GRU는 RNN으로 통칭한다.


## output과 h_n의 차이

* **output**

> shape: (seq_len, Batch_size, D * hidden_size) (Batch_first=False일 시)

`output`은 *모든 time step*의 *final layer*의 hidden state들이다. 즉 주어진 문장 내 모든 토큰들의 마지막 hidden state이다. 만약 양방향 RNN이면 각 토큰별 순방향과 역방향시 h_t시 concat되어 리턴되므로 마지막 차원의 크기는 2 * `hidden_size`이다.

* **h_n**

> shape: (D * n_layers, Batch_size, hidden_size)

`h_n`는 *마지막 time step (token)*의 *모든 layer*의 hidden state이다. `D=2` 즉 양뱡향 RNN 모델일 때 `h_n`의 짝수번째 벡터는 순방향, 홀수번째 벡터는 역방향 layer의 최종 결과물이다. 

|          |    output    | h_n |
|----------|:------------:|:---:|
| layer    |  final only  | all |
| timestep | 1, 2, ..., t |  t  |

## output과 h_n의 관계

레이어 수가 하나인 단방향 RNN의 경우에만 `h_n`은 `output`의 부분집합이다. 정확히 말해 마지막 토큰의 벡터인 `output[-1]`이 `h_n`이다. 그러나 레이어 수가 2 이상이고, 양뱡향인 RNN의 경우 `output`과 `h_n`의 정확한 관계를 파악하기가 어렵다.



```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# pad_token의 id: 0
X = [[12, 23, 1, 16, 59, 6, 37],
     [6,  3,  2, 76, 31, 0, 0],
     [12, 6,  1,  0,  0, 0, 0],
     [11, 21, 18, 1,  0, 0, 0]]

X = torch.tensor(X, dtype=torch.long)
X.shape # (batch_size, seq_len)
```




    torch.Size([4, 7])




```python
# pad_token을 제외한 실제 token들의 len
input_lens = torch.tensor([torch.max(row.nonzero())+1 for row in X])
input_lens, indices = input_lens.sort(descending=True)
X_sorted = X[indices]

print(f'정렬 후: {X_sorted}')
print(f'패딩 토큰을 제외한 실토큰 개수: {input_lens}')

```

    정렬 후: tensor([[12, 23,  1, 16, 59,  6, 37],
            [ 6,  3,  2, 76, 31,  0,  0],
            [11, 21, 18,  1,  0,  0,  0],
            [12,  6,  1,  0,  0,  0,  0]])
    패딩 토큰을 제외한 실토큰 개수: tensor([7, 5, 4, 3])



```python
hidden_size=256

class RNNModel(nn.Module):
  def __init__(self, rnn_type):
    super(RNNModel, self).__init__()

    self.embedding = nn.Embedding(100, 128)
    self.num_layers = 2
    self.hidden_size = hidden_size

    self.rnn = getattr(nn, rnn_type)(
        input_size=128, 
        hidden_size=self.hidden_size,
        num_layers=2,
        bidirectional=True,
    )

  def forward(self, Batch, Batch_lens):  # Batch: (batch_size, seq_len), Batch_lens: (batch_size)
    # d_w: word emBedding size
    batch_emb = self.embedding(Batch)  # (batch_size, seq_len, d_w)
    batch_emb = batch_emb.transpose(0, 1)  # (seq_len, batch_size, d_w)

    packed_input = pack_padded_sequence(batch_emb, Batch_lens)
    h_0 = torch.zeros((self.num_layers * 2, Batch.shape[0], self.hidden_size))  # (num_layers*num_dirs, batch_size, d_h) = (4, batch_size, d_h)
    packed_output, h_n = self.rnn(packed_input, h_0)  # h_n: (4, batch_size, d_h)
    output = pad_packed_sequence(packed_output)[0]  # outputs: (seq_len, batch_size, 2*d_h)

    return output, h_n, packed_output
```


```python
# 양방향 RNN이고, batch_size = False일 때

rnn = RNNModel(rnn_type='GRU')
output, h_n, packed_outputs = rnn(X_sorted, input_lens)
print(f'output shape (seq_len, batch_size, 2*hidden_size) == {output.shape}')
print(f'h_n shape (2*num_layers, batch_size, hidden_size) == {h_n.shape}')

```

    output shape (seq_len, batch_size, 2*hidden_size) == torch.Size([7, 4, 512])
    h_n shape (2*num_layers, batch_size, hidden_size) == torch.Size([4, 4, 256])


* **순방향**

순방향시 마지막 time-step은 문장 맨 마지막 토큰이며, 그 값은 `output` 마지막 차원의 첫 `hidden_size`개의 원소와 같다 (순방향+역방향을 붙인 벡터이기 때문에).

* **역방향**

역방향시에는 마지막 time-step이 문장 맨 처음 토큰이며, 그 값은 `output` 마지막 차원내 뒤에서부터 `hidden_size`개의 원소와 같다.


```python
# forward: last time-step (i.e. last token), last-layer
print(torch.eq(output[-1, 0, :hidden_size], h_n[2, 0, :]).all())

# backward: last time-step (i.e. first token), last-layer
print(torch.eq(output[0, 0, hidden_size:], h_n[3, 0, :]).all())
```

    tensor(True)
    tensor(True)


`output`과 `h_n`의 관계를 일반화 해보자.


```python
# backward 
batch_size = output.shape[1]
for batch_idx in range(batch_size):
    if torch.eq(output[0, batch_idx, hidden_size:], h_n[3, batch_idx, :]).all().item():
        print(f'At {batch_idx}th batch output & h_n --> equal')
```

    At 0th batch output & h_n --> equal
    At 1th batch output & h_n --> equal
    At 2th batch output & h_n --> equal
    At 3th batch output & h_n --> equal


역방향 rnn layer의 경우 `output`의 첫번째 토큰(`0`)은 모든 배치에서 `[hidden_size : ]` 원소들과 `h_n[3]`의 원소들은 일치한다 (앞서 말했듯 `h_n`은 모든 레이어의 마지막 time-step의 hidden_state를 리턴하는데, 첫 레이어의 순방향, 첫 레이어의 역방향, 두번째 레이어의 순방향, 두번째 레이어의 역방향시 마지막 hidden state들을 차례차례 리턴한다).  

순방향의 rnn layer는 어떨까?


```python
# forward
for batch_idx in range(batch_size):
    if torch.eq(output[-1, batch_idx, :hidden_size], h_n[2, batch_idx, :]).all().item():
        print(f'At {batch_idx}th batch output & h_n --> equal')
    else:
        print(f'At {batch_idx}th batch output & h_n --> NOT equal')

```

    At 0th batch output & h_n --> equal
    At 1th batch output & h_n --> NOT equal
    At 2th batch output & h_n --> NOT equal
    At 3th batch output & h_n --> NOT equal


마지막 토큰의 hidden state 즉 `output[-1, batch_idx, :hidden_size]`를 했음에도 몇몇 배치에서는 예상과 다르게 나왔다. 문제를 알아보기 위해 다음 코드를 돌려봤다.


```python
# forward
for batch_idx in range(batch_size):
    for token_idx in range(output.shape[0]):
        if torch.eq(output[token_idx, batch_idx, :hidden_size], h_n[2, batch_idx, :]).all().item():
            print(f'For {token_idx}th token at {batch_idx}th batch output & h_n --> equal')

```

    For 6th token at 0th batch output & h_n --> equal
    For 4th token at 1th batch output & h_n --> equal
    For 3th token at 2th batch output & h_n --> equal
    For 2th token at 3th batch output & h_n --> equal


각 배치마다 `output`과 `h_n`이 겹치는 토큰 인덱스가 다른데, 정렬한 인풋의 (패딩 토큰을 제외한) 실토큰의 개수와 일치함을 알 수 있다. 이는 마지막 time step의 hidden state라고 하는 `h_n`의 경우 (패딩 토큰이 없는) 실제 문장의 마지막 토큰의 hidden state을 담기 때문이다.


```python
# 위에서 만든 정렬한 인풋
print(f'정렬 후: {X_sorted}')
print(f'패딩 토큰을 제외한 실토큰 개수: {input_lens}')
```

    정렬 후: tensor([[12, 23,  1, 16, 59,  6, 37],
            [ 6,  3,  2, 76, 31,  0,  0],
            [11, 21, 18,  1,  0,  0,  0],
            [12,  6,  1,  0,  0,  0,  0]])
    패딩 토큰을 제외한 실토큰 개수: tensor([7, 5, 4, 3])

