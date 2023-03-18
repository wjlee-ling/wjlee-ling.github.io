---
title: Bias-Variance Trade-off Revisited
description: 딥러닝에서도 bias-variance trade-off는 불가피한가?
search: true
toc: true
date: 2023-03-18
categories:
    - TIL
tags: 
    - ML
---

면접 스터디를 하다가 머신러닝의 기본 중에 기본인 bias-variance trade-off 얘기가 나왔다. 그런데 갑자기 든 생각이 최신 딥러닝 모델들도 이 trade-off를 갖고 있는지? 였다. 무엇보다 이전에 케라스를 만든 사람이 쓴 트윗인가 아니면 저서 <케라스 창시자에게 배우는 딥러닝>에서 이 trade-off를 극복할 수 있다는 식의 주장을 봤던 기억이 있다 (정확하지 않음). 그래서 잠깐 구글링 해보니 최근에도 이에 대한 [논문 A Modern Take on the Bias-Variance Tradeoff in Neural Networks](https://arxiv.org/abs/1810.08591)이 있어서. 궁금증을 풀겸 bias-variance trade-off에 대해 정리해 본다.


## bias-variance trade-off

오랜 시간 머신러닝의 진리(?)로 받아드려진 bias-variance trade-off의 정의를 [영문 위키피디아](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)에서 찾아보았다.

> In statistics and machine learning, the bias–variance tradeoff is the property of a model that the variance of the parameter estimated across samples can be reduced by increasing the bias in the estimated parameters. The bias–variance dilemma or bias–variance problem is the conflict in trying to simultaneously minimize these two sources of error that prevent supervised learning algorithms from generalizing beyond their training set.

직접 번역해 보면
> 통계와 머신러닝에서 편향-분산 트레이드오프는 표본에 따라 예측된(훈련된) 파라미터의 분산이 해당 파라미터의 편향(오차)이 증가할 때 감소할 수 있는 모델의 속성이다. 이 편향-분산 딜레마 또는 문제라고도 불리는 이슈는 교사 학습 알고리즘(모델)이 학습셋 외에도 일반화하는 걸 막는 두 종류의 오류, 즉 편향과 분산을 동시에 최소화하려는 과정에서 발생하는 충돌이다.

여기서 편향(bias)과 분산(variance)을 보다 자세히 알아보자. 

* 참고
1. 위키
2. [코넬 강의자료](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote12.html)
3. [Gentle Introduction to the bias-variance trade-off in Machine Learning](https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/)

### bias
> The bias error is an error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).

편향(bias)은 모델의 예측값과 실제 값과의 오차이다. 편향이 크다는 것은 간단히 말해 모델이 해당 셋에 대해 제대로 못 맞춘다는 것으로 모델의 **underfitting**을 보여준다. 그렇다고 해서 훈련셋의 examples을 최대한 맞추는 식으로 모델을 학습해 이 오차를 줄이면 정작 다른 셋에서는 성능이 떨어질 수 있다.

### variance
> Variance captures how much your classifier changes if you train on a different training set. How "over-specialized" is your classifier to a particular training set (overfitting)? If we have the best possible model for our training data, how far off are we from the average classifier?

분산(variance)은 예측하려는 셋에 따라 보이는 모델의 예측값의 분산이다. 일반적인 머신러닝 과제에 대입해 생각해보면, 훈련셋에 학습된 모델의 분포가 검증셋에서 예측한 분포와 얼마나 다른지를 나타내는 것이다. 사실 나는 이 트레이드오프에서 말하는 분산이 셋마다 달라지는 예측값의 분산인지 아니면 동일 셋에서의 예측값의 분산인지 헷갈렸는데, 전자를 말한다. 
결국 분산이 크다는 것은 모델이 훈련셋과 검증셋에서 보이는 분포가 많이 다르다는 뜻으로, 모델이 훈련셋에 있는 노이즈나 아웃라이어에 지나치게 민감할 때 **overfitting**이 나타날 수 있다.

즉 편향-분산 트레이드오프는 훈련셋에서의 예측력과 검증셋에서의 일반화 사이의 줄다리기이다.

## 