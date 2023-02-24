# Word2Vec Implementation
> "Word2Vec SGNS(Skip-Gram Negative Sampling) 구현하기"

- toc: true
- badges: true
- comments: true
- date: 2022-06-10
- last-modified-at: 2022-06-22
- categories: [TIL, cs224n]

Pytorch로 Word2Vec의 SGNS 구현해 보았다.

* reference:
1. [SGNS 논문](https://arxiv.org/pdf/1310.4546.pdf)  (Mikolov et al. 2013)
2. [word2vec official notebook](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/word2vec.ipynb)

https://www.jasonosajima.com/ns.html
http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/


```python
#hide 
from google.colab import drive
drive.mount('/content/drive')
!pip install sentencepiece
!pip install tqdm
!pip install pytorch_lightning
!pip install gensim
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: sentencepiece in /usr/local/lib/python3.7/dist-packages (0.1.96)
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (4.64.0)
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: pytorch_lightning in /usr/local/lib/python3.7/dist-packages (1.6.4)
    Requirement already satisfied: tensorboard>=2.2.0 in /usr/local/lib/python3.7/dist-packages (from pytorch_lightning) (2.8.0)
    Requirement already satisfied: fsspec[http]!=2021.06.0,>=2021.05.0 in /usr/local/lib/python3.7/dist-packages (from pytorch_lightning) (2022.5.0)
    Requirement already satisfied: pyDeprecate>=0.3.1 in /usr/local/lib/python3.7/dist-packages (from pytorch_lightning) (0.3.2)
    Requirement already satisfied: torch>=1.8.* in /usr/local/lib/python3.7/dist-packages (from pytorch_lightning) (1.11.0+cu113)
    Requirement already satisfied: packaging>=17.0 in /usr/local/lib/python3.7/dist-packages (from pytorch_lightning) (21.3)
    Requirement already satisfied: tqdm>=4.57.0 in /usr/local/lib/python3.7/dist-packages (from pytorch_lightning) (4.64.0)
    Requirement already satisfied: numpy>=1.17.2 in /usr/local/lib/python3.7/dist-packages (from pytorch_lightning) (1.21.6)
    Requirement already satisfied: typing-extensions>=4.0.0 in /usr/local/lib/python3.7/dist-packages (from pytorch_lightning) (4.1.1)
    Requirement already satisfied: protobuf<=3.20.1 in /usr/local/lib/python3.7/dist-packages (from pytorch_lightning) (3.17.3)
    Requirement already satisfied: torchmetrics>=0.4.1 in /usr/local/lib/python3.7/dist-packages (from pytorch_lightning) (0.9.1)
    Requirement already satisfied: PyYAML>=5.4 in /usr/local/lib/python3.7/dist-packages (from pytorch_lightning) (6.0)
    Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from fsspec[http]!=2021.06.0,>=2021.05.0->pytorch_lightning) (2.23.0)
    Requirement already satisfied: aiohttp in /usr/local/lib/python3.7/dist-packages (from fsspec[http]!=2021.06.0,>=2021.05.0->pytorch_lightning) (3.8.1)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging>=17.0->pytorch_lightning) (3.0.9)
    Requirement already satisfied: six>=1.9 in /usr/local/lib/python3.7/dist-packages (from protobuf<=3.20.1->pytorch_lightning) (1.15.0)
    Requirement already satisfied: setuptools>=41.0.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch_lightning) (57.4.0)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch_lightning) (1.8.1)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch_lightning) (0.4.6)
    Requirement already satisfied: wheel>=0.26 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch_lightning) (0.37.1)
    Requirement already satisfied: google-auth<3,>=1.6.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch_lightning) (1.35.0)
    Requirement already satisfied: grpcio>=1.24.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch_lightning) (1.46.3)
    Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch_lightning) (3.3.7)
    Requirement already satisfied: absl-py>=0.4 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch_lightning) (1.1.0)
    Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch_lightning) (0.6.1)
    Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch_lightning) (1.0.1)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard>=2.2.0->pytorch_lightning) (4.2.4)
    Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard>=2.2.0->pytorch_lightning) (4.8)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard>=2.2.0->pytorch_lightning) (0.2.8)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard>=2.2.0->pytorch_lightning) (1.3.1)
    Requirement already satisfied: importlib-metadata>=4.4 in /usr/local/lib/python3.7/dist-packages (from markdown>=2.6.8->tensorboard>=2.2.0->pytorch_lightning) (4.11.4)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard>=2.2.0->pytorch_lightning) (3.8.0)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.7/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard>=2.2.0->pytorch_lightning) (0.4.8)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch_lightning) (1.24.3)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch_lightning) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch_lightning) (2022.6.15)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch_lightning) (3.0.4)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard>=2.2.0->pytorch_lightning) (3.2.0)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch_lightning) (1.7.2)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.7/dist-packages (from aiohttp->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch_lightning) (1.3.0)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.7/dist-packages (from aiohttp->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch_lightning) (1.2.0)
    Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /usr/local/lib/python3.7/dist-packages (from aiohttp->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch_lightning) (4.0.2)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch_lightning) (21.4.0)
    Requirement already satisfied: asynctest==0.13.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch_lightning) (0.13.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.7/dist-packages (from aiohttp->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch_lightning) (6.0.2)
    Requirement already satisfied: charset-normalizer<3.0,>=2.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch_lightning) (2.0.12)
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: gensim in /usr/local/lib/python3.7/dist-packages (3.6.0)
    Requirement already satisfied: six>=1.5.0 in /usr/local/lib/python3.7/dist-packages (from gensim) (1.15.0)
    Requirement already satisfied: smart-open>=1.2.1 in /usr/local/lib/python3.7/dist-packages (from gensim) (5.2.1)
    Requirement already satisfied: scipy>=0.18.1 in /usr/local/lib/python3.7/dist-packages (from gensim) (1.4.1)
    Requirement already satisfied: numpy>=1.11.3 in /usr/local/lib/python3.7/dist-packages (from gensim) (1.21.6)



```python
import os
import pandas as pd
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

main_dir = r"/content/drive/MyDrive/Colab Notebooks/nlp/a2" # 이 notebook이 저장된 폴더 주소
os.chdir(main_dir) 
np.random.seed(42)
```

## Corpus & Tokenization 

[SentencePiece](https://github.com/google/sentencepiece)로 Unigram Wordpiece로 한국어 word2vec를 진행했다. 훈련 데이터는 모두의말뭉치-신문기사(2021, 조선일보, 140MB)의 일부이다. 사전 크기는 5,000으로 정했으며, 추후 사용할 수 있게끔 special tokens (e.g. [PAD], [SEP], etc)도 사전에 등록하였다 (훈련은 안함).

* 참고:
1. https://github.com/google/sentencepiece/tree/master/python
2. https://github.com/paul-hyun/transformer-evolution/blob/master/tutorial/vocab_with_sentencepiece.ipynb


```python
corpus = pd.read_csv('/content/drive/MyDrive/A2_TeamProject/data/cleaned_모두의말뭉치/모두의말뭉치-조선일보.csv')
corpus.groupby('topic').nunique()
```





  <div id="df-dc2d9793-007b-4fce-9ece-26dedca80b86">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>newpaper</th>
      <th>title</th>
      <th>body</th>
      <th>written_at</th>
    </tr>
    <tr>
      <th>topic</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>IT/과학</th>
      <td>1</td>
      <td>2937</td>
      <td>19457</td>
      <td>605</td>
    </tr>
    <tr>
      <th>경제</th>
      <td>1</td>
      <td>5530</td>
      <td>35929</td>
      <td>668</td>
    </tr>
    <tr>
      <th>문화</th>
      <td>1</td>
      <td>1690</td>
      <td>8345</td>
      <td>541</td>
    </tr>
    <tr>
      <th>미용/건강</th>
      <td>1</td>
      <td>2569</td>
      <td>17585</td>
      <td>578</td>
    </tr>
    <tr>
      <th>사회</th>
      <td>1</td>
      <td>13362</td>
      <td>82599</td>
      <td>710</td>
    </tr>
    <tr>
      <th>생활</th>
      <td>1</td>
      <td>4831</td>
      <td>31584</td>
      <td>677</td>
    </tr>
    <tr>
      <th>스포츠</th>
      <td>1</td>
      <td>2983</td>
      <td>18621</td>
      <td>625</td>
    </tr>
    <tr>
      <th>연예</th>
      <td>1</td>
      <td>326</td>
      <td>1837</td>
      <td>251</td>
    </tr>
    <tr>
      <th>정치</th>
      <td>1</td>
      <td>4344</td>
      <td>24469</td>
      <td>631</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-dc2d9793-007b-4fce-9ece-26dedca80b86')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-dc2d9793-007b-4fce-9ece-26dedca80b86 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-dc2d9793-007b-4fce-9ece-26dedca80b86');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




다양한 범주의 본문을 random하게 추출하여 학습데이터를 만들었고, 구두점이나 기호들을 제거했다. 구두점/기호들은 대개 문맥적으로 의미가 떨어지는 한편, 그 수가 너무 많아 오버샘플링 된다.


```python
# 소장한 data format에 맞게 전처리 진행 
indices = np.random.randint(0, len(corpus), 10000)
corpus = corpus['body'][indices].tolist() #기사의 본문만 추출해 사용.
sentences = []
for paragraph in corpus:
    sents = paragraph.split('.') # 사용한 corpus의 단위가 paragraph 기반이라 sentence기반으로 바꿈.
    for sent in sents:
        if sent == '':
            continue
        sent = sent.strip()
        for punc in ".,!?\"'_-$^&*(){}[]<>/#+=\\:;": 
            sent = sent.replace(punc, '')
        sentences.append(sent) 
corpus_text = '\n'.join(sentences) # SentencePiece는 input으로 one-sentence-per-line 요구함
# 확인
for sent in sentences[:3]:
    print(sent)

with open('KorNews.txt', 'w', encoding='utf-8') as f:
    f.write(corpus_text)
```

    선택권을 가진 미위팅이 백번을 선택했다
    요즘 고수들의 초반 포석은 마치 복기를 구경하는 것 같다
    패턴화한 10여 개 남짓의 정석이 반상에 반복적으로 주르르 펼쳐진다


Word2Vec은 원래 word 단위로 학습되었고, 두 어절 이상의 단어(예시 'White House')들 중 몇 개는 따로 사전에 등록해서 하나의 단위로 학습하게끔 했다. 

그러나 고립어인 영어와 달리, 한국어는 교착어로 조사/어미가 복잡하고 어절 단위로 사전을 구성하기에 부적합하기 때문에 여기서는 SentencePiece를 사용하여 단어 보다 작은 subword 단위로 토큰화하여 사전을 구성하려고 한다.


```python
vocab_size = 10000
spm.SentencePieceTrainer.train(input='KorNews.txt', model_prefix='KorNews_Unigram', vocab_size=vocab_size, model_type='unigram',\
                               pad_id=0, pad_piece="[PAD]", unk_id=1, unk_piece="[UNK]", bos_id=2, bos_piece="[BOS]", eos_id=3, eos_piece="[EOS]",\
                               user_defined_symbols=["[SEP]", "[CLS]", "[MASK"])
```


```python
# hide
## optional: 이미 만들어 놓은 corpus와 spm model 사용할 때
try:
    assert vocab_size
except:
    vocab_size = 10000

with open(os.path.join(main_dir, 'KorNews.txt'), 'r', encoding='utf-8') as f: 
    corpus_text = f.readlines()

sentences = [sent.strip('\n') for sent in corpus_text]
```


```python
# 위 train 과정으로 KorNews.model 과 KorNews.vocab 이 저장됨.
sp = spm.SentencePieceProcessor()
sp.load(os.path.join(main_dir, "KorNews_Unigram.model"))
vocab = {sp.id_to_piece(id): id for id in range(sp.get_piece_size())}
```


```python
tokens = []
for sent in sentences:
    encoded = sp.encode(sent, out_type=int, add_bos=False, add_eos=False)
    tokens.append(encoded)
```

## Skip-Gram Negative Sampling


**Error Distribution**

논문에서 negative sampling에 필요한 error distribution을 구할 때 다음과 같은 확률을 이용한다. 
$$
P_n(w_i) = \frac{U(w_i)^{3/4}}{\sum_{j=0}^{n}U(w_j)^{3/4}}
$$
$U(w_i)$는 unigram probability이다.

위 공식으로 각 단어가 추출될 확률을 구해 ``probs``라는 변수에 저장한다. 해당 변수는 위에서 지정한 word2id의 인덱스를 따른다. (즉 [UNK], [SOS], [EOS] 토큰은 각각 0,1,2의 인덱스 가짐). 이 확률을 이용하여 negative example의 사전 id를 구하려면 
1. python 내장 random 라이브러리의 choices 함수를 사용하거나 
2. numpy.random.choice를 이용하면 된다 (이 경우에는 k만큼 반복)
3. [matrix 사용하는 방법](https://ethankoch.medium.com/incredibly-fast-random-sampling-in-python-baf154bd836a). 위 두 방법이 가장 간단하나 시간이 너무 오래 걸려 찾아 보았다. 이 방식은 1/2번 보다 훨씬 빠르나, 내 경우엔 vocab_size가 너무 커서 (vocab_size, vocab_size)만큼 생성할 때 부담.

```python
import random, numpy
indices = list(range(vocab_size))
ids = random.choice(indices, weigths=probs, k) # 1번
ids = [numpy.random.choice(indices, p=probs) for _ in range(k)] # 2번
```
윗 방법들이 시간/물리적으로 부담이 크기에 gensim의 [word2vec 코드](https://github.com/RaRe-Technologies/gensim/blob/ded78776284ad7b55b6626191eaa8dcea0dd3db0/gensim/models/word2vec.py#L822)**Word2Vec.make_cum_table()**를 보고 참고한 sampling 방법이 아래 **Skipgram._create_cumsum_table()**과 **Skipgram.sample_negatives()**이다. 이 방법은 위 공식의 $P(w_i)$를 누적시킨 확률 array를 만들어 놓고, [0,1] 사이의 난수를 bisection-left방식 (np.searchsorted 참고)으로 누적 확률 array에 비교시켜 결과로 얻은 인덱스를 샘플링할 단어의 인덱스로 여기는 것이다. 
예를 들어 사전 크기가 3이고 각 단어들의 샘플링 확률(0.75승 하여 완만하게 만든 확률)이 [0.1, 0.3, 0.6]일 때, cumsum_table은 [0.1, 0.4(=0.1+0.3), 1.0(=0.1+0.3+0.6)]이다. 생성한 난수가 0.28일 때 이 난수와 cumsum_table로 bisection search(left)를 하면 이 난수가 속할 section은 두번째 섹션(b/c 0.1 < **0.28** <= 0.4)이므로 negative sample로 추출할 단어 인덱스는 1이다.  

**Subsampling Probability**
$$
P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}}
$$
논문에서는 빈도가 매우 높은 단어의 경우 오버샘플링 되어 빈도수가 적은 단어들보다 훨씬 업데이트가 될 가능성이 높은 만큼, 각 단어의 샘플링 추출 확률을 정해 훈련했다고 한다. 위 확률은 추출되지 않을 확률이다. t는 임의로 정하는 값이며, 논문에서는 1e-5로 했으나 나는 1e-4로 했다 (gensim은 (0,1e-5]의 값).


```python
from collections import Counter

n_neg_samples = 10 # 'k' in the paper; # of negative samples per each pair (center word, positive context word)
window_size = 2 # i.e. 2*2 context words. 원래는 center + 앞/뒤 context words를 다 포함하는 크기

class SkipgramSampler():
    def __init__(self, tokens, vocab):
        '''
        Args:
            - tokens: List[List]. Each nested list should account for a sentence.
            - vocab: Dict{word:id}. Word2id
        '''
        self.tokens = tokens
        self.vocab_size = len(vocab) # word2id
        print(f'vocab_size: {self.vocab_size}')
        self.vocab = vocab
        self.id2word = {id: w for w, id in vocab.items()}

    def _get_freq(self):
        '''
        get frequency of words
        '''
        counter = Counter(self.id2word.keys())
        # update counter
        for sent_tokens in self.tokens:
            counter.update(sent_tokens)
        # calculate pure counts
        counts = []
        for id in self.id2word.keys():
            counts.append(counter[id])
        counts = np.array(counts, dtype=float)
        freq = counts / counts.sum()
        return freq

    def _get_negative_sampling_prob(self, freq):
        probs = freq**0.75
        probs /= probs.sum()           
        return probs

    def _get_sub_sampling_prob(self, freq, t=3e-5):
        '''
        논문의 공식과 다르게 버려질 확률이 아닌 뽑힐 확률. 논문에서는 t=1e-5
        '''
        probs = np.sqrt(t / freq) #-np.sqrt(probs**-1 * 1e-5 ) + 1
        return probs

    def _create_cumsum_table(self, freq):
        '''
        vocab 인덱스를 따름. 단어별 sampling 축적 확률.
        '''
        probs = self._get_negative_sampling_prob(freq)
        cumsum_table = np.cumsum(probs)
        assert cumsum_table[-1] - 1. <= 1e-5
        return cumsum_table

    def sample_skipgrams(self, window_size=2, k=5, return_ids=True):
        '''
        Args:
            - window_size: 2*window_size context words for each target word
            - k: num of the negative pairs per each true context word
            - return_ids : id로 출력 혹은 word(str)로 출력
        Return:
            - pairs: [center_word, positive_context_word, negative_context_word_1, ..., negative_context_word_k]
        '''
        pairs, labels = [], []
        freq = self._get_freq()
        subsampling_probs = self._get_sub_sampling_prob(freq)
        negsampling_probs = self._get_negative_sampling_prob(freq)
        cumsum_table = self._create_cumsum_table(negsampling_probs)
        discarded, saved = 0, 0
        for sent in tqdm(self.tokens):
            sent_len = len(sent)

            for idx, center_id in enumerate(sent):
                idx_start = idx-window_size if idx-window_size >= 0 else 0
                idx_end = idx+window_size+1 if idx+window_size+1 < sent_len else sent_len
                window = sent[idx_start:idx_end]
                pos_ids = [id for id in window if id != center_id]
                random_probs = np.random.random(len(pos_ids))
                pos_sub_probs = np.array([subsampling_probs[id] for id in pos_ids])
                pos_sub_probs = (random_probs <= pos_sub_probs) # if to be sampled, True

                for pos_id, pos_sub_prob in zip(pos_ids, pos_sub_probs):
                    if not pos_sub_prob:
                        discarded += 1
                        continue
                    saved += 1
                    neg_ids = self.sample_negatives(k, cumsum_table, [center_id]+ pos_ids)

                    if return_ids:
                        pairs.append([center_id, pos_id])
                        labels.append(1)
                        for neg_id in neg_ids:
                            pairs.append([center_id, neg_id])
                            labels.append(0)
                    else:
                        pairs.append([self.id2word[center_id], self.id2word[pos_id]])
                        labels.append(1)
                        for w in [self.id2word[id] for id in neg_ids]:
                            pairs.append([self.id2word[center_id], w])
                            labels.append(0)

        print(f'\nsaved: {saved} discarded: {discarded}')
        return pairs, labels

    def sample_negatives(self, k, cumsum_table, exceptions):
        '''
        unique하게 negaitve samples를 추출. center word와 true context words (i.e.exceptions)는 제거.
        '''
        neg_ids = []
        exceptions = np.array(exceptions)
        while len(neg_ids) < k:
            values = np.random.random(k-len(neg_ids))
            #ids = self.random.choices(list(range(self.vocab_size)), weights=probabilities, k=k-len(neg_ids))
            ids = np.searchsorted(cumsum_table, values, side='left')
            ids = np.setdiff1d(ids, exceptions) #unique하게 됨
            neg_ids.extend(ids)
        return neg_ids

ss = SkipgramSampler(tokens, vocab)
pairs, labels = ss.sample_skipgrams(window_size=window_size, k=n_neg_samples, return_ids=True)
print(f'positive examples: {labels.count(1)} & negative examples: {labels.count(0)}')
```

    vocab_size: 10000


    100%|██████████| 35174/35174 [01:26<00:00, 405.61it/s]


    
    saved: 1340766 discarded: 2283114
    positive examples: 1340766 & negative examples: 13407660


총 10,000 문장에서 positive: 1,340,766 단어쌍, negative: 13,407,660 단어쌍을 추출하였다.

오랜 시간 걸려 추출한 n-gram samples을 저장하려면 
```python3
import json

# write
pairs = {'center_words': centers, 'context_words':contexts}
with open(os.path.join(main_dir, 'KorNews_SGNS_Pairs.json'), 'w') as f:
    json.dump(pairs, f)

# read
json_dir = os.path.join(main_dir, "KorNews_SGNS_Pairs.json")
with open(json_dir, 'r') as f:
    pairs_dict = json.load(f)
# pairs_dict['center_words']
```


```python
from torch.utils.data import Dataset, DataLoader
import torch

class SKNS_Dataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx][0], self.pairs[idx][1], self.labels[idx]

dataset = SKNS_Dataset(pairs,labels)
```


```python
test_size = int(len(dataset) * 0.05)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - test_size, test_size])
train_dataloader = DataLoader(dataset, shuffle=True, batch_size=64, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=4)
del dataset
```

## SKNS 모델

> Note: 항상 pytorch를 사용했었는데, 이번엔 좀 더 빠르고 가볍다는 [pytorch-lightning]("https://github.com/PyTorchLightning/pytorch-lightning")을 이용해 훈련해보기로 했다. 기존에는 1) model 클래스와 인스턴스 만들고 dataloader-loop 안에서 2) loss계산과 back-propagation 3)checkpoint saving 을 하는 코드를 일일히 따로 작성해야 한다. 그런데  라이트닝에서는 model 클래스만 지정해주면 제공되는 Trainer를 이용해 간단히 훈련시킬 수 있다.  

Word2Vec은 사전 인덱스를 이용한 one-hot embedding이 아니라, 각 단어마다 300차원의 벡터를 생성하고 훈련시킨다. dataloader로 사전 인덱스를 배치로 받기 때문에 해당 인덱스에 해당하는 n-차원의 벡터로 바꿔주는 nn.Embedding layer를 사용한다. Word2Vec은 같은 단어가 센터(타겟)냐 컨텍스트냐에 따라 다른 벡터를 갖기 때문에 (vocab_size, embedding_size)의 임베딩 텐서를 두 개 생성해야 한다.



```python
from torch import optim, nn
import pytorch_lightning as pl

embedding_dim = 256 # 논문은 300 

center_vecs = nn.Embedding(vocab_size, embedding_dim)
context_vecs = nn.Embedding(vocab_size, embedding_dim)

class SKNS(pl.LightningModule):
    def __init__(self, center_vecs, context_vecs):
        super().__init__()
        self.centers = center_vecs
        self.contexts = context_vecs
        self.embed_size = self.centers.weight.shape[1]
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, center_ids, context_ids):
        # in lightning, forward defines the prediction/inference actions
        batch_size = len(center_ids)
        V = self.centers(center_ids).view(batch_size, self.embed_size, 1) # (batch, embedding) -> (batch, embed, 1)
        U = self.contexts(context_ids).view(batch_size, 1, self.embed_size) # (batch, embedding) -> (batch, 1, embed)
        logits = torch.bmm(U, V).squeeze() # (batch, 1, 1) => (batch)
        return logits

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        center_ids, context_ids, labels = batch
        labels = labels.float()
        logits = self(center_ids, context_ids)
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        center_ids, context_ids, labels = batch
        labels = labels.float()
        logits = self(center_ids, context_ids)
        loss = self.loss_fn(logits, labels)
        self.log("valid_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch:0.95 ** epoch)
        return {
            "optimizer":optimizer,
            "lr_scheduler":scheduler
        }

    def get_weights(self):
        return (self.centers.weight, self.contexts.weight)

model = SKNS(center_vecs, context_vecs)
trainer = pl.Trainer(accelerator='gpu', max_epochs=5, precision=16)
```

    Using 16bit native Automatic Mixed Precision (AMP)
    GPU available: True, used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs


> Note: loss 함수로 쓴 nn.BCEWithLogitsLoss() 는 binary classification에 쓸 수 있는 함수로 log(simoid(x)) 한 loss값을 계산해준다.label의 dtype은 int가 아닌 float이여야 한다.


```python
pl.utilities.memory.garbage_collection_cuda()

trainer.fit(model, train_dataloader, test_dataloader) # DataLoader 객체가 아닌 따른 Iterator여도 됨.
```

    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    
      | Name     | Type              | Params
    -----------------------------------------------
    0 | centers  | Embedding         | 2.6 M 
    1 | contexts | Embedding         | 2.6 M 
    2 | loss_fn  | BCEWithLogitsLoss | 0     
    -----------------------------------------------
    5.1 M     Trainable params
    0         Non-trainable params
    5.1 M     Total params
    10.240    Total estimated model params size (MB)



    Sanity Checking: 0it [00:00, ?it/s]



    Training: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



```python
%reload_ext tensorboard
%tensorboard --logdir=lightning_logs/
```


    Output hidden; open in https://colab.research.google.com to view.


## 모델 평가

gensim을 이용하여 유의어/반의어 및 analogy test를 할 수 있다.


```python
import gensim
(centers, contexts) = model.get_weights()

with open('KorNews_w2v_V.txt', 'w') as f:
    f.write(f'{vocab_size-1} {embedding_dim}\n')
    for w, ind in vocab.items():
        f.write(f'{w} {" ".join(map(str, centers[ind,:].detach().cpu().tolist() ))}\n')
w2v = gensim.models.KeyedVectors.load_word2vec_format('KorNews_w2v_V.txt')
```

결과를 보면 어느 정도 학습이 된 것으로 보인다. 그러나 코퍼스의 출처가 작년 신문 기사다 보니 정치 관련 단어들은 비교적 학습이 잘 된 반면, 일반명사나 동사, 부사 등은 학습이 아쉬워 보인다. 게다가 subword 토크나이저는 대개 어절의 맨 앞에 나타나는 경우('▁중국')와 아닌 경우('중국')로 나눠 사전을 구성하는데, 두 경우다 제대로 훈련시키는 것도 해결해야할 과제이다.


```python
w2v.most_similar(positive=['▁중국'])
```




    [('▁한국', 0.41443902254104614),
     ('▁미국', 0.4098540246486664),
     ('▁일본', 0.38627344369888306),
     ('▁세계', 0.3704119026660919),
     ('▁20', 0.34667396545410156),
     ('▁선수', 0.34340745210647583),
     ('▁7', 0.3106424808502197),
     ('▁트럼프', 0.3065921664237976),
     ('▁국내', 0.3052994906902313),
     ('▁우리', 0.30426502227783203)]




```python
w2v.most_similar(positive=['중국'])
```




    [('▁영', 0.27694588899612427),
     ('마케팅', 0.2600131928920746),
     ('▁기자들', 0.2560494542121887),
     ('▁샘플', 0.24935927987098694),
     ('▁밖에', 0.24410425126552582),
     ('홍보관', 0.24130752682685852),
     ('▁신한카드', 0.2389327883720398),
     ('벼', 0.2308148890733719),
     ('셧', 0.22932805120944977),
     ('▁의견이', 0.22786958515644073)]




```python
w2v.most_similar(positive=['▁대통령'])
```




    [('▁청와대', 0.3450790047645569),
     ('▁장관', 0.3396446704864502),
     ('▁의원', 0.32314038276672363),
     ('▁이후', 0.3140028119087219),
     ('▁대통령이', 0.3101692497730255),
     ('▁민주당', 0.30879759788513184),
     ('▁정부', 0.3069238066673279),
     ('▁문', 0.3028714060783386),
     ('엔', 0.29196697473526),
     ('▁미국', 0.28609851002693176)]




```python
w2v.most_similar(['▁트럼프'])
```




    [('▁미', 0.358543336391449),
     ('▁대통령이', 0.3487488031387329),
     ('▁북한', 0.3468424677848816),
     ('▁일본', 0.31102779507637024),
     ('▁미국', 0.31074368953704834),
     ('▁중국', 0.3065921664237976),
     ('▁지난달', 0.2924591600894928),
     ('▁총리', 0.2919672429561615),
     ('▁대통령', 0.2834343910217285),
     ('▁검찰', 0.26758497953414917)]




```python
w2v.most_similar(positive=['▁내년'])
```




    [('▁지난해', 0.2932113707065582),
     ('▁9', 0.2712252736091614),
     ('▁이날', 0.2705169916152954),
     ('▁온라인', 0.2697073817253113),
     ('▁올해', 0.2610678970813751),
     ('▁2021', 0.2584304213523865),
     ('▁AP', 0.25618451833724976),
     ('▁감염자', 0.2545149326324463),
     ('학년도', 0.24879765510559082),
     ('▁이어', 0.24502748250961304)]




```python
w2v.most_similar(positive=['을'])
```




    [('를', 0.4446668028831482),
     ('으로', 0.3875734508037567),
     ('이', 0.3176490068435669),
     ('고', 0.2827107608318329),
     ('도', 0.25626111030578613),
     ('은', 0.254934698343277),
     ('에', 0.25474870204925537),
     ('▁것', 0.23995378613471985),
     ('하는', 0.2388738989830017),
     ('과', 0.2370939552783966)]




```python
w2v.most_similar(['한다'])
```




    [('했다', 0.48733481764793396),
     ('할', 0.44905075430870056),
     ('하는', 0.4311229884624481),
     ('하고', 0.4263955354690552),
     ('해', 0.4139285683631897),
     ('됐다', 0.40646255016326904),
     ('하기', 0.3861447274684906),
     ('될', 0.3859519362449646),
     ('돼', 0.3652912378311157),
     ('하면서', 0.3618863821029663)]




```python
w2v.most_similar(['▁군'])
```




    [('산', 0.3136850595474243),
     ('▁다른', 0.2931358218193054),
     ('▁주', 0.29162198305130005),
     ('▁지난해', 0.28887131810188293),
     ('▁청와대', 0.2820741534233093),
     ('▁장', 0.2817824184894562),
     ('▁문제', 0.27845048904418945),
     ('▁20', 0.27788224816322327),
     ('▁기', 0.2763606011867523),
     ('▁대해', 0.26905715465545654)]



### 수정사항

1. 더 많은 학습데이터로 추가 학습 (문장 10,000개 선정하여 사전 만들고 및 학습하였음.)
2. 혹은 사전 크기를 줄이기
