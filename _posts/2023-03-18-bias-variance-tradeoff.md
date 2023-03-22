---
title: Bias-Variance Trade-off, Revisited
description: Double Descent
search: true
toc: true
date: 2023-03-18
categories:
    - TIL
tags: 
    - bias-variance trade-off
    - double descent
---

면접 스터디를 하다가 머신러닝의 기본 중에 기본인 bias-variance trade-off 얘기가 나왔다. 그런데 갑자기 든 생각이 최신 딥러닝 모델들도 이 trade-off를 갖고 있는지? 였다. 무엇보다 이전에 케라스를 만든 사람이 쓴 트윗인가 아니면 저서 <케라스 창시자에게 배우는 딥러닝>에서 이 trade-off를 극복할 수 있다는 식의 주장을 봤던 기억이 있다 (정확하지 않음). 그래서 구글링 해보니 최근에 이에 대한 논의가 여럿 있어서 궁금증을 풀겸 bias-variance trade-off에 대해 정리해 본다.


## ML 모델의 bias-variance trade-off

오랜 시간 머신러닝의 진리(라고 번역해야 할지... literature에서 tenet이라 하는 것)로 받아드려진 bias-variance trade-off의 정의를 [영문 위키피디아](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)에서 찾아보았다.

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

<figure>
    <img width="1000" alt="bias and variance error graph" src="https://user-images.githubusercontent.com/61496071/226149026-c07689ad-dda2-4da9-8e3f-cb72a29b529b.png">
    <figcaption style="text-align: center">bias-variance trade-off에 따르면 모델의 복잡도가 올라갈수록 bias는 줄어들지만 variance가 증가한다.</figcaption>
</figure>

## DL 모델의 bias-variance trade-off
내 궁금증대로 최신 딥러닝 모델이 bias-variance trade-off를 극복할 수 있다는 주장들이 있었다.


### double descent

**Double Descent**에 대한 [openAI의 deep double descent 설명](https://openai.com/research/deep-double-descent)을 번역해 보면,

> 모델과 데이터의 크기, 훈련 시간을 늘렸을 때 모델의 성능이 처음엔 나아지다가, 악화되고, 다시 개선되는 *double descent 현상*이 CNN, ResNet, 트랜스포머 모델에서 발생한다는 것을 확인했다. 이 현상은 적절한 규제(regularization)로 막을 수 있다. 이러한 현상이 모델 상관없이 전반적으로(universially) 나타나는데, 왜 발생하는지 제대로 파악하지 못했으며 중요한 연구점으로 판단하고 있다. 

[Reconciling modern machine learning practice and the bias-variance trade-off](https://arxiv.org/pdf/1812.11118.pdf)는 큰 딥러닝 모델이 큰 데이터셋으로 훈련이 됐을 때 bias-variance trade-off 공리대로 test error가 전통적인 U-shape을 그리다가 interpolation(즉 훈련 셋에서 loss가 0) 지점이 지나면 다시 둘 다 줄어든다고 주장한다.

> When we increase the function class capacity high enough (e.g., by increasing the number of features or the size of the neural network architecture), the learned predictors achieve (near) perfect fits to the training data—i.e., interpolation. Although the learned predictors obtained at the interpolation threshold typically have high risk, we show that **increasing the function class capacity beyond this point leads to decreasing risk**, typically going below the risk achieved at the sweet spot in the “classical” regime.

<figure>
    <img width="1000" alt="double descnet" src="https://user-images.githubusercontent.com/61496071/226151417-3849c74c-71d3-4e6b-83ee-b4c9f74886eb.png">
    <figcaption style="text-align: center;">training risk(error)가 0이 되는 interpolation threshold 이후 test error가 줄어든다.</figcaption>
</figure>

그렇다고 해서 항상 "큰 (딥러닝) 모델이 항상 옳다!!"로 귀결할 수는 없다. interpolation point까지 도달할 데이터와 모델 크기를 현실적으로 확보하는 게 어렵기 때문이다. 아래 그림에서 보듯이 모델이 애매하게 크면 bias-variance trade-off대로 오히려 error가 올라간다. 

<figure>
    <img width="1000" alt="critical regim" src="https://user-images.githubusercontent.com/61496071/226152239-867e5017-4c8d-47b4-91d7-716bf50df3ca.png">
    <figcaption style="text-align: center;">출처: OpenAI</figcaption>
</figure>

모델의 크기 뿐만 아니라 데이터의 크기도 고심해야 한다. 

<figure>
    <img width="1000" alt="more data could hurt" src="https://user-images.githubusercontent.com/61496071/226152389-c90b796b-ecf7-4566-b0a2-214b0e492778.png">
    <figcaption style="text-align: center;">출처: OpenAI</figcaption>
</figure>

위 그림은 번역 과제에서 데이터의 크기와 트랜스포머 모델의 크기에 따라 loss값이 어떻게 되는지를 보여주는데, 오히려 데이터가 많을 때 적을 때보다 성능이 떨어지는 모델 사이즈 구간이 있음을 보여준다. 즉 무작정 데이터를 늘린다고 능사가 아니라, 데이터 크기와 모델 크기 사이의 밸런스를 고려해야 한다는 뜻이다.
