---
title: 정규화?
search: true
toc: true
description: normalization, regularization, standardization 차이 이해하기
date: 2022-11-21
categories:
    - TIL
tags: 
    - normalization
    - regularization
    - standardization
---

머신러닝, 딥러닝계에서 normalization, standardization, regularization은 대개 정규화라는 용어로 번역이 되기 때문에 개인적으로 각각의 개념과 쓰임이 헷갈리기도 했다. 그래서 한번 정리해 봤다.

## Normalization

**Reference**
1. [Google의 normalization 설명](https://developers.google.com/machine-learning/data-prep/transform/normalization?hl=en)
2. [Codecademy normalization](https://www.codecademy.com/article/normalization)


**Normalization**은 그 자체로 다양한 통계학적 기법을 포함하는 용어이다. 좁은 의미로는 min-max scaling이며 이때는 z-score scaling을 의미하는 **standardization**과 비교할 수 있다 (혼용하기도 한다: [scikit-learn 예시](https://scikit-learn.org/stable/modules/preprocessing.html)). 반면 큰 의미로는 z-score standardization의 상위 개념으로, 데이터 값을 다시 스케일(*rescale*)하는 기법들을 일컫는다. 

데이터 값의 범위를 조정하는 이유는 높은 feature들마다 비슷한 스케일 범위를 갖게 하기 위함이다. 몇몇의 feature가 다른 feature들보다 값의 범위가 훨씬 클 때 데이터 분포가 전자의 차원들 위주로 형성하게 되고 모델의 훈련과 성능이 떨어지게 된다. 

구글의 설명을 바탕으로 협의와 광의의 normalization 기법 4가지, min-max scaling, clipping, standardization, log scaling을 정리해보자.

### min-max scaling

(min-max) **Normalization**은 다음의 식으로 값들의 범위를 [0, 1]로 조정한다.

$$ x' = \frac{x - x_{min}}{x_{max} - x_{min}} $$

다만 임금처럼 최댓값과 최솟값의 차이가 엄청 큰 특성에 대해서는 min-max scaling이 위험할 수도 있다고 한다 (구글 포스트 참고). 예를 들어 말단 사원의 임금과 (이에 수천배 더 벌 수 있는) CEO의 임금을 일괄적으로 정규화하면 소수의 고임금자들 때문에 임금 분포가 극단적으로 좌편향되기 때문이다. 즉 min-max 정규화는 아웃 라이어 처리에 효과적이지 않다.


### Clipping

**Clipping**은 데이터의 분포를 말 그대로 정해진 범위로 잘라내 범위를 벗어나는 값들은 범위의 극값으로 지정하는 방법이다. clipping을 사용해 극단적 값을 가지는 outlier를 정리한 후 다른 정규화 방법을 사용할 수 있다. 이 기법이 RNN에서 역전파되는 gradient가 폭발(exploding)하거나 0에 수렴(vanishing)하는 것을 막는 **gradient clipping**외에도 머신러닝/딥러닝계에서 사용되는지는 모르겠다.

### log scaling

**Log scaling**은 log-scale로 분포를 완만하게 바꿔준다. 이 방식은 데이터의 분포가 편향이 심할 때 (예시: 임금 분포) 사용하면 좋다. 대신 음수의 데이터는 처리할 수 없다.

## Standardization

좁은 의미의 **Standardization**(표준화)는 z-score normalizationo을 의미하며 데이터 분포가 표준정규분포, 즉 평균이 0 표준편차가 1이 되도록 만든다. centering과 scaling을 차례로 적용하며 그 수식은 다음과 같다.

$$ x' = \frac{x-\mu}{\sigma}$$

표준화의 장점은 학습을 빠르게 하고 안정적으로 하는 것이다. MNIST 데이터로 딥러닝 모델을 만든다고 하자. 데이터를 표준화하지 않으면 첫 히든 레이어는 인풋으로 [0, 255] 범위의 값들을 받아 활성화 함수로 [0, 1] (sigmoid 사용 시), [-1, 1] (tanh 사용 시) 사이의 값들로 출력하기 때문에 이후 레이어들보다 대략 255배 높은 값들을 처리하게 된다. 그 결과 첫 레이어의 활성화 함수에서 saturation이 일어나 역전파시 업데이트가 거의 안돼 모델 훈련을 느리게 할 수 있다.
 
표준화도 min-max scaling처럼 outlier가 많을 때는 분포가 크게 왜곡될 수 있다. 

최근 SOTA 모델들은 대개 레이어들이 처리하는 값의 범위를 일정하게 만들어주는 **batch normalization**이나 **layer normalization**을 사용하여 학습을 안정화한다. 딥러닝에서 매우 중요한 개념이므로 이에 관한 포스트는 따로 정리해야겠다.

## Regularzatin

**Reference**
1. [Deep Learning Book Ch. 7](https://www.deeplearningbook.org/contents/regularization.html)
2. [Regularization Method: Noise for improving Deep Learning models](https://towardsdatascience.com/noise-its-not-always-annoying-1bd5f0f240f)
3. [Normalization과 Regularization](https://gaussian37.github.io/dl-concept-regularization/)
4. [Regularizing Deep Neural Networks by Noise: Its Interpretation and Optimization](https://proceedings.neurips.cc/paper/2017/file/217e342fc01668b10cb1188d40d3370e-Paper.pdf)

**Regularization**은 모델이 훈련 셋에 과적합(overfitting)되는 것을 막아 일반화를 하는 작업들을 일컫는다. 똑같이 ‘정규화’로 번역되는 standardization/normalization과 구분해 ‘규제’라고 번역되기도 한다. 사실 모든 딥러닝 모델들이 일반화를 추구하는 만큼 다양한 규제법이 있어, 이 역시도 따로 정리해야 할 필요가 있다. Regularization은 크게 보면 weight의 크기를 줄이거나 노이즈를 일부러 추가하거나 훈련을 다양하게 할 수 있다. 또는 단순히 훈련 데이터를 늘리면 일반화 오차(generalization error)를 줄일 수 있다. 

### 가중치(Weight) 규제

[과적합이 이뤄지면 가중치들도 크기 때문에](https://stats.stackexchange.com/questions/64208/why-do-overfitted-models-tend-to-have-large-coefficients) 가중치들의 합(놈; norm)을 비용함수에 추가하여, 비용함수를 줄이는 최적화 과정에서 가중치도 작게 만들어 과적합을 막을 수 있다. 이렇게 추가적인 페널티로 쓰이는 놈은 벡터의 크기로 $${\lVert{x}\rVert}_p = {(\sum_{k=1}^{n}{|x_k|}^p)}^{1/p}$$이다. p 값에 따라 비용 함수가 L1, L2 loss가 되는데 L1 놈은 추론에 영향력이 작은 차원의 가중치를 0에 가깝게 줄이기 때문에 feature를 가려내는 데(*feature selection*) 도움이 된다고 하고, L2 놈은 가중치를 골고루 줄인다고 한다. 

### 노이즈 추가

노이즈가 추가해 훈련을 하면 훈련이 더욱 어려워지기 때문에 과적합을 막을 수 있다. 노이즈는 히든 유닛에 추가하거나 인풋 데이터에 더할 수 있다. **드랍아웃**은 대표적인 규제법으로 주어진 확률로 히든 유닛을 끄고(i.e. 해당 가중치를 0으로 하고) 훈련시키는 방법이다. 미니 배치마다 마스킹을 달리해 순전파를 하기 때문에 사실상 앙상블의 효과를 가진다고 한다. 드랍아웃 외에 가중치에다가 노이즈를 추가하는 방법도 가능하다고 한다.

히든 유닛 말고 데이터 자체에 노이즈를 추가하거나 마스킹을 추가하는 방법도 있다. 후자의 방법은 사실상 드랍아웃과 동일한 게 아닌가 싶다 ($ y = WX $ 에서 $W$를 조작하면 드랍아웃, $X$를 건들이면 마스킹?)

노이즈를 추가하는 것은 훈련을 방해하여 과적합을 막는 한편 정반대로 훈련할 데이터 수를 늘려 일반화 성능을 높인다.
