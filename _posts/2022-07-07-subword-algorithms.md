---
title: subword tokenization 알고리즘
search: true
toc: true
description: BPE, WordPiece, Unigram, SentencePiece 정리
date: 2022-07-07
categories:
    - TIL
tags: 
    - BPE
---
## Subword Tokenization

Subword Tokenization이란 띄어쓰기(space)를 기준으로 나뉘는 단어보다 작지만 character(자/모)보다 큰 유닛(subword)으로 문장을 나누는 것으로 다음과 같은 장점이 있다. 
1. 빈도가 낮은 단어, 사전에 없는 단어들도 (빈도가 높은) 서브 단어들의 조합으로 인코딩할 수 있다.
2. 따라서 적당한 크기의 사전으로 많은 단어를 커버할 수 있다.
3. 띄어쓰기를 안하는 언어(예. 중국어, 일본어) 처리에 용이하다.
4. 접사/어미 등이 실질적 의미를 갖는 어근, 어간에 달라 붙어 어절을 형성하는 교착어인 한국어 처리에 보다 용이하다.

> ❗️ 그렇다고 해서 하나의 형태소를 하나의 subword로 취급하는 것이 아니다. 

BPE의 다양한 알고리즘을 [huggingface의 설명](https://huggingface.co/docs/transformers/tokenizer_summary#bytepair-encoding-bpe)을 참고해 정리해봤다.

## 기본 아이디어

**훈련 방식**

0. 정규화
1. pre-tokenization (어절 단위로 자르기) & 기본 사전 만들기: 문장을 단어 단위로 나누기. 띄어쓰기 위주의 토큰화(e.g. GPT-2, Roberta 경우)도 가능하지만 보다 복잡한 토큰화를 사용할 수도 있다(GPT, XLM 등). 얻어진 토큰들을 기본 사전으로 삼고, 각 토큰들의 빈도수를 센다.
2. 위에서 구한 기본 사전들의 각 유닛들의 조합들 중 가장 빈도가 높은 조합을 사전에 추가한다.
3. 사전에 정한 크기에 사전이 될 때까지 (2)를 반복한다. 

**예시**

(1) 사용하는 corpus가 pre-tokenization 이후 다음과 같은 토큰들과 빈도수를 갖는다고 가정할 때
```
frquency = [("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)]
vocabulary = ['b, 'g', 'h', 'n', 'p', 's', 'u']
>> [("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)]
```
(2) 사전에 등록된 아이템 조합 중 'u'+'g'가 가장 빈도가 높으므로 'ug'를 사전에 추가한다. 
```
("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)
>> vocabulary = ['b, 'g', 'h', 'n', 'p', 's', 'u', 'ug'] # update
```
(3) 정해진 사전 크기까지 '조합 + 사전 등록' 반복. 

> note: `</w>`와 같은 특별 토큰을 단어 끝에 붙여 단어간 경계를 표기하고 훈련시키기도 한다.

## BPE(Byte Pair Encoding) 

위 예시에서 보듯이 BPE는 이름대로 빈도가 높은 바이트쌍들을 파악하여 사전에 새 유닛으로 추가한다. 그래서 어절 (또는 띄어쓰기) 단위로 사전을 만들 때 크기가 12라고 한다면, 빈도가 높은 sub-string 2쌍을 새로 사전에 추가하면 크기를 10으로 줄일 수 있다.
```
b, g, h, p, n, s, u, hug, pug, pun, bun, hugs  -> b, g, h, p, n, s, u, ug, un
``` 

### BBPE (Byte-level Byte Pair Encoding)
알파벳 외에도 한글이나 이모지가 있을 때에는 사전을 어떻게 만들까? 'ㄱ'부터 '힣'까지 한글에 있는 모든 캐릭터로만 사전을 만든다고 해도 크기가 10,000을 넘긴다. 이러한 단점을 극복하고자 BBPE는 유니코드를 바이트로 바꿔 BPE를 적용한다. 바이트 기반으로 기본 사전을 만들기 때문에 256(==2^8)이라는 작은 크기의 기본 사전으로 모든 sub-string쌍을 커버할 수 있는데 예를 들어 GPT-2는 1) 기본 256 바이트 유닛 2) 이들을 결합해 만든 50,000개의 조합과 \<eos>이라는 스페셜 토큰을 추가해 총 50,257짜리 사전을 구성한다.

**문제점**
빈도수가 똑같은 서브워드 쌍(pair)들이 여러 있을 때 어떤 쌍을 우선시할지 애매하다. 추가되는 쌍에 따라 같은 단어가 여러가지 방법으로 다르게 인코딩 될 수 있으며, 이는 최종 성능 평가에 영향을 줄 수 있다.

## WordPiece

BPE와 기본 아이디어와 훈련 방식이 거의 비슷한 알고리즘이 여러 있다. BERT등이 사용하는 WordPiece도 그 중 하나인데, 사전에 추가하는 기준이 살짝 다르다. BPE는 단순 빈도로 평가했다면, WordPiece는 가능도(likelihood)를 따진다. 또 위 BPE는 바이트들을 결합시켜 새 유닛을 만드는 bottom-up 방식인 반면, WordPiece는 bottom-up은 물론 top-down 방식으로도 구현할 수 있다(한국어, 일본어, 중국어 등은 top-down은 안됨).

WordPiece를 소개한 [논문](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf)을 보면 language model을 만들고 bigram으로 가능도를 측정한다고 되어있다. huggingface [설명](https://huggingface.co/course/chapter6/6?fw=pt)에 따르면 이 가능도는 $$\frac{C(w_n)}{C(w_{n-1}) \times C(w_n)}$$ 로 계산한다. 직관적으로 이해하자면, 각 단어의 빈도가 적을수록(분모) + 둘이 같이 나타나는 빈도가 높을수록(분자) 결합 순위가 높아진다는 것이다. 다만 huggingface측이 제시한 수식이 일반적으로 n-gram language model에서 maximum likelihood 구하는 수식이랑 달라 (분모에서 C(w_n) 추가로 곱하는 것) 조금 의문이 생기는데, huggingface도 구글이 코드를 공개하지 않아 추측해서 만들었다고 한다.
> Warning: "Google never open-sourced its implementation of the training algorithm of WordPiece, so what follows is our best guess based on the published literature. It may not be 100% accurate." (huggingface)

원 논문의 알고리즘은 bottom-up인데, BERT에 사용되었다는 top-down WordPiece 알고리즘은 BPE처럼 단순 빈도를 따지고 pair 구성도 bigram이상으로 하는 등 여러모로 독특하다. [여기](https://www.tensorflow.org/text/guide/subwords_tokenizer#optional_the_algorithm) 참고. 같은 character라도 prefix가 있는 경우('h + ug' vs. 'h + '##ug')와 없는 경우 다르게 encoding하는 것도 원 논문에는 없지만 일반적으로 많이 구현하는 방식인듯 하다.

## Unigram

Unigram은 BPE나 WordPiece와 확연히 다르게 큰 사전으로 시작해서, loss 상 없어져도 큰 상관이 없는 토큰을 없애는 방식으로 사전의 크기를 조정한다. 얼마나 상관이 있는지 계산하는 방식은 이름 그대로 unigramg하게 정한다. 예를 들어 corpus가 다음과 같은 (subword, count)로 이뤄져 있다면 가능한 subword 조합은 다음과 같다.
```
("dog", 10) ("do", 5) ==> ("d", 15) ("o", 15) ("g", 10) ("do", 15) ("og", 10)
```
이를 바탕으로 unigram 확률을 구하자면

$$ 
P(["d", "o", "g"]) = P("d") \cdot P("o") \cdot P("g") = \frac{15}{65} \frac{15}{65} \frac{10}{65} = 0.008192990441511151 \\
P(["do", "g"]) = P("do") \cdot P("g") = \frac{15}{65} \frac{10}{65} = 0.03550295857988166 \\
P(["d", "og"]) = P("d") \cdot P("og") = \frac{15}{65} \frac{10}{65} = 0.03550295857988166
$$

이다. 즉 서브워드 토큰의 unigram 확률의 분모가 조합가능한 모든 페어의 빈도를 합한 값이므로 최대한 적은 수의 서브워드로 나누는 것이 확률값이 크다. 이와 같이 unigram상 확률의 negative log likelihood로 훈련 코퍼스에 대한 총 loss를 구하게 된다.

$$
\mathcal{L} = - \log \prod_{t}^{T}{P(w_t)} = - \sum_{t}^{T}{\log P(w_t)} 
$$

인데 모든 subword pair 들 중 편의상 세 조합("o", "og", "do")만을 골라 loss를 비교해 보면,
1) "o"라는 subword를 없앤 후 loss 

$$ \mathcal{L} = - \sum_{t}^{T}{\log P(w_t)} = - 10 \cdot \log P(["do", "g"]) - 5 \cdot \log P(["do"]) = 40.713077800917326 $$

2)  "og"라는 subword를 없앤 후 loss

$$ \mathcal{L} = - \sum_{t}^{T}{\log P(w_t)} = - 10 \cdot \log P(["do", "g"]) - 5 \cdot \log P(["do"]) = 40.713077800917326 $$

"o" 또는 "og"를 제외한 조합으로 총 loss를 계산했을 때 두 loss간의 차이는 없다 (애초에 "do"로 분할하기 때문에). 그러나

3) "do"라는 subword pair를 사전에서 없애면

$$ \mathcal{L} = - 10 \cdot \log P(["d", "og"]) - 5 \cdot \log P(["d", "o"]) = 48.044763144884456 $$

오히려 loss가 증가한다. 따라서 세 후보 중 없앴을 때 loss가 최소로 증가하는 pair인 "o" 나 "og"를 사전에서 제거하게 된다.

## SentencePiece
[**SentencePiece**](https://github.com/google/sentencepiece)는 별도의 알고리즘이 아니라 BPE와 Unigram 알고리즘을 지원하는 라이브러리로 구글이 공개했다. SentencePiece가 사용하는 BPE는 살짝 다르다고 한다.

## Reference
1. huggingface: [BPE Tokenizer Summary](https://huggingface.co/docs/transformers/tokenizer_summary#bytepair-encoding-bpe)
2. huggingface: [BPE Tokenization](https://huggingface.co/course/chapter6/5?fw=pt#bytepair-encoding-tokenization)
3. google-tensorflow: [Subword Tokenization](https://www.tensorflow.org/text/guide/subwords_tokenizer#optional_the_algorithm)
4. huggingface: [Byte-Pair Encoding](https://huggingface.co/course/chapter6/5?fw=pt)
5. huggingface: [WordPiece](https://huggingface.co/course/chapter6/6?fw=pt)
6. huggingface: [Unigram](https://huggingface.co/course/chapter6/7?fw=pt)
