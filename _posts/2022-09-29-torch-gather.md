---
title: torch.gather 직관적으로 이해하기
search: true
description: dim & index, 어떻게 할 것인가
date: 2022-09-29
categories:
    - TIL
tags: 
    - torch
---
얼마 전 torch에서 말하는 dimension에 대해 다루는 포스트에서 gather 함수를 예시로 다뤄보았다. 사실 2차원을 대상으로 torch.gather을 적용하는 것은 직관적으로 이해가 되나, 3차원 이상 텐서에 적용하려면 `dim` 값은 어떻게 해야할지, `index`는 어떻게 정해야할지 이해가 쉽지 않다. 긴 시간 고민해 보고 나름대로 정리한 내용을 공유한다.

## 기본 조건 이해하기

pytorch의 [torch.gather](https://pytorch.org/docs/stable/generated/torch.gather.html#torch.gather)의 설명을 보면

```
Gathers values along an axis specified by dim.

For a 3-D tensor the output is specified by:
    out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
    out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

input and index must have the same number of dimensions. It is also required that index.size(d) <= input.size(d) for all dimensions d != dim. out will have the same shape as index. Note that input and index do not broadcast against each other.

```
위 설명을 요약하자면

1. `index`와 `input`은 차원의 수(**rank**)가 같아야 한다 (`input`이 3차원이면 `index`도 3차원, 2차원이면 2차원)
2. `index`와 `out`은 크기(shape)가 같아야 한다 (따라서 당연히 차원의 수도 같다)

위 두 성질을 합치면 `input`이 3차원이면, `index`도 3차원이여야 하고 `out` 텐서도 3차원이다.

## 핵심은 dim

`index`의 형식은 `input`과 원하는 `out`의 형식에 의해 미리 정해진다고 했다. 진짜로 신경써야 할 점은 `index` tensor에 채워 넣을 내용으로, 이 때 `dim`이 중요하다. 

다시 pytorch의 설명을 빌려오자면
```out[i] [j] [k] = input[index[i][j][k]] [j] [k]  # if dim == 0 ```이다. 내가 지정해 주는 `dim` 차원을 제외한 나머지 차원의 원소들은 `input`과 동일하다는 것이다. 즉 `index`의 `dim`차원을 *변수*로, 나머지 차원들을 *상수*로 두고 `index`를 채워넣어야 한다.

3차원의 텐서에서 대각선에 위치한 원소들만 뽑아 2차원 행렬을 만들어 보자. 

![problem-solution](/assets/images/torch-gather-example1.png)



```python
import torch 

_input = torch.tensor([i for i in range(1, 13)])
_input = _input.view(2, 2, 3)
_input

```




    tensor([[[ 1,  2,  3],
             [ 4,  5,  6]],
    
            [[ 7,  8,  9],
             [10, 11, 12]]])



### dim=1 일 때

`input`이 3차원이니까 `index`도 3차원의 텐서로 넣어줘야 한다. 그리고 인덱스를 추출할 차원`dim`이 1이기 때문에 *row-wise*로 인덱싱을 해야한다.

> Note: 2차원 행렬에서는 dim=0이 row-wise, dim=1이 column-wise이지만, 배치 차원이 추가된 3차원 텐서에서는 dim=0이 batch-wise, dim=1이 row-wise, dim=2가 column-wise이다 (일반적인 파이토치 기준. 데이터에 따라 차원 구성은 다름.)


`index[0][0][0]`에 배정할 값을 `input`에서 찾는다고 할 때, 1차원만 변수고 0차원(배치)과 2차원(열)이 모두 0으로 고정이므로 `index[0][0][0] = input[0][?][0]`이다. 물음표에 들어갈 인덱스를 찾는게 문제인데, 이를 그림으로 표현하면 다음과 같다.

![gather-dim1-1-ex](/assets/images/gather-dim1-ex1.png)

위 `input` 텐서에서 우리가 찾는 값은 1이고, 이 값은 해당 열(노란색)의 0번 인덱스 값이다. 따라서 `index[0][0][0] = 0`이다. 

다음으로 `index[0][0][1]`에 들어갈 값을 찾아보자. 이번엔 0차원이 0, 2차원이 1로 고정이므로 `index[0][0][1] = input[0][?][1]`이고, ?에 들어갈 값은 그림에서 1번 열(녹색)에서 찾아야 한다. 녹색 열에서 우리가 찾는 값은 대각선에 있는 5이고, 해당 열에서 5의 인덱스는 1이므로 `index[0][0][1] = 1`이다.

이런 식으로 `index[0][0][2]`를 구하려고 했더니, 애초에 우리가 원하는 `out`에는 `out[0][0][2]`값이 없다. 사실 우리가 0번 배치에서 뽑아낼 값은 다 뽑아냈으므로 더 이상 인덱싱을 할 필요가 없고, 따라서 `index[0]`은 다 채웠다.

이제 `index[1][0][0]`의 값을 찾으려면 `input[1][?][0]`, 즉 1번째 배치의 첫 열(청색)을 봐야 하고 우리가 찾는 값은 0번째 인덱스에 있으므로 `index[1][0][0] = 0`이다.

마찬가지로 `index[1][0][1]`에 들어갈 값은 동일 배치내 두번째 열(주황색)의 1번 인덱스에 있으므로 `index[1][0][1] = 1`이다.


```python
index = torch.tensor([[[0, 1]],
                    [[0, 1]]])
                    
out = torch.gather(_input, dim=1, index=index)
out
```




    tensor([[[ 1,  5]],
    
            [[ 7, 11]]])



앞서 말했듯 (1) `index`의 차원 수와 `input`의 차원 수가 일치해야 하고 (2) `index`의 shape과 `out`의 `index`가 동일해야 하므로 결과물도 3차원이 나왔다. 원래 3차원의 텐서의 대각선에 있는 값들을 2차원의 행렬로 모아주는 것이 문제이므로 차원을 줄여준다.


```python
# shape 조정하기
out = out.squeeze() # 또는 view 사용
out
```




    tensor([[ 1,  5],
            [ 7, 11]])



### dim = 2 일 때

이번엔 인덱싱을 column-wise로 해보자. 2번째 차원, 즉 feature 차원이 *변수*이고 batch/observation 차원이 *상수*이다 그림으로 인덱싱 작업을 표현하면 다음과 같다. 

![gather-dim2-ex](/assets/images/gather-dim2-ex1.png)

우리가 추출할 값들은 배치마다 0번째 행의 0번 인덱스, 1번째 행의 1번 인덱스이다. 


```python
index = torch.tensor([ [[0],
                        [1]],
                        [[0],
                        [1]] ])
                    
out = torch.gather(_input, dim=2, index=index)
out
```




    tensor([[[ 1],
             [ 5]],
    
            [[ 7],
             [11]]])



우리가 원하는 `out` 모양이 실제 결과물과 다르므로 추가 작업을 해준다. 


```python
out = out.view(2,2)
out
```




    tensor([[ 1,  5],
            [ 7, 11]])



### dim = 0 일 때

dim=0, 즉 배치 차원을 변수로 두고도 인덱싱이 가능할까? 한 큐에는 불가능하다. 인덱싱을 할 행차원과 열차원이 *상수*이기 때문이다. 

예를 들어 추출해야 할 값 5와 11은 각각 0번째, 1번째 배치의 [1,1]에 위치하고 있고 따라서 `index` 텐서의 `[...][1][1]` 인덱스가 존재해야 한다. 그런데 어떤 텐서가 `[...][1][1]` 인덱스를 갖고 있다면, 최소 (2,2)의 shape를 가진 텐서여야 하고 당연히 인덱스 `[...][0][1]`, `[...][1][0]`에 값이 존재해야 한다. 그런데 우리가 추출할 값들(1, 5, 7, 11)은 행/열의 인덱스가 [0][0] 이거나 [1][1]이므로 `index[...][0][1]`이나 `index[...][1][0]`은 존재할 수 없다.

## 정리 및 결론

`index`와 `out`의 shape이 같아야 하고, 지정해 주는 `dim` 차원에 따라 인덱싱을 한다는 점을 명심하자. 또 `dim`이 아닌 차원들은 상수로서 `index`와 `input`의 그것과 일치한다는 점을 기억하자. 

따라서 (그나마) 직관적으로 `index`와 `dim`을 선택할 팁이라면 우선 원하는 output의 구조(shape)대로 `index`의 구조를 머릿속에 그려놓고, 어떤 `dim`으로 인덱스를 찾으면 편할지 생각해 보자. 이렇게 `dim`을 정한 후 우리가 원하는 값들이 지정한 `dim`의 어떤 인덱스에 있는지 역으로 확인하는 식으로 찾으면 된다.


