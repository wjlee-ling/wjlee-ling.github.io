---
title: pytorch 훅(hook)
search: true
description: 
date: 2022-09-27
categories:
    - TIL
tags: 
    - torch
---

## hook 이란

> A **hook** is a place and usually an interface provided in packaged code that allows a programmer to insert customized programming.

패키지 사용자가 직접 만든 커스텀 코드를 사용할 수 있게끔 개발자가 만들어 놓은 인터페이스.

PyTorch에서도 hook을 사용할 수 있는데, `Tensor`와 `nn.Module`에 적용할 수 있다. Module에는 forward, backward hook을 사용할 수 있는 반면, Tensor 타입은 backward hook만 지원한다. 이름대로 forward/backward hook들은 각각 순전파, 역전파시 작동한다. 또한 `nn.Module`에는 pre-hook이라고 `.forward()`를 돌기 전에 적용되는 hook도 있다.

| | nn.Module| Tensor|
|---|---|---|
|forward|     O |   X|
|backward|    O |   O|
|pre-hook| O | X|

`nn.Module`에 등록된 hook들은 model.__dict__에서 확인할 수 있다.

## 용법

1. 디버깅

모델을 만들 때, 각 서브모듈별로 인풋과 아웃풋(또는 웨이트)의 크기가 맞지 않아 에러가 발생하거나 제대로 작동하는지 확인해야 할 때가 많다. 이럴 때 흔히 일일히 `print`문을 달아서 코드를 고쳐야 하는데, pre_hook 또는 hook을 사용하면 한큐에 해결할 수 있다.

2. 중간 과정 값 저장하기

딥러닝 모델은 흔히 여러 개의 층으로 이뤄져 있다. 일반적으로 모델을 돌려 우리가 얻는 결과는 말 그대로 모든 층을 거친, 마지막 층의 값들이다. 만약 중간 층의 output이 필요하다면 hook을 이용해 저장하여, 추후 분석에 사용할 수 있다.

3. gradient clipping

gradient clipping은 exploiding gradient을 막고자 threshold를 넘기는 gradient 값을 조정하는 것이다. backward hook을 사용하여 threshold를 넘기는 로컬 gradient에 clipping을 적용할 수 있다. Hook을 사용하지 않고도 텐서의 ```grad```에 접근할 수 있지만, 이때는 모든 층의 역전파가 끝난 후에만 가능하다.


## Reference
1. [How to Use PyTorch Hooks](https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904)
2. [PyTorch 101, Part 5: Understanding Hooks](https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/)
