# Part 1. Multimodal Learning
## Chapter 1. Foundations of Multimodal Learning
### 1.1 Why Multimodal Learning?
현실 세계의 데이터는 본질적으로 하나의 표현 방식만으로 주어지지 않는다. 인간은 시각, 언어, 청각, 촉각 등 다양한 감각을 통합하여 세상을 이해하며, 실제 데이터 문제도 유사한 구조를 가진다. 예를 들어 의료에서는 MRI나 CT 같은 영상 정보, EMR 텍스트, 검사 수치, 유전체 정보가 함께 사용될 수 있고, 자율주행에서는 카메라, LiDAR, Radar, GPS, 텍스트 지도 정보가 동시에 활용될 수 있다. 이처럼 서로 다른 표현 형식의 데이터 단위를 **modality**라고 한다.

Multimodal learning은 이러한 여러 modality를 함께 사용하여 단일 modality보다 더 풍부하고 강건한 표현을 학습하는 문제를 다룬다. 핵심은 단순 결합이 아니라, 서로 다른 정보원이 제공하는 상보적 정보와 공통 구조를 효과적으로 활용하는 데 있다. 따라서 이 분야의 핵심 질문은 다음과 같다.

- 서로 다른 modality를 어떤 공간에서 정렬할 것인가
- 어떤 시점에 어떤 방식으로 결합할 것인가
- 일부 modality가 없을 때도 어떻게 강건하게 동작하게 할 것인가
- 학습된 표현을 분류, 회귀, 검색, 생성, 추론에 어떻게 활용할 것인가

### 1.2 What Is a Modality?
modality는 데이터가 관측되거나 표현되는 방식이다. 예를 들어 다음은 서로 다른 modality의 예시이다.
- image: 픽셀 격자나 patch token으로 표현
- text: token sequence로 표현
- audio: waveform 또는 spectrogram으로 표현
- time series: 시간에 따라 변화하는 signal sequence로 표현
- tabular data: 고정 길이 feature vector로 표현
- graph: node-edge 구조로 표현

즉, multimodal learning은 단순히 입력 개수가 많은 문제가 아니라, **서로 구조와 통계적 성질이 다른 입력을 함께 다루는 문제**라고 볼 수 있다.

### 1.3 Basic Formulation
총 $M$개의 modality가 있다고 하자.
```math
x^{(1)}, x^{(2)}, \dots, x^{(M)}
```
여기서 각 기호의 의미는 다음과 같다.
- $M$: modality의 개수
- $x^{(m)}$: $m$번째 modality의 입력 데이터
- 예를 들어 $x^{(1)}$은 이미지, $x^{(2)}$는 텍스트, $x^{(3)}$는 음성일 수 있다
멀티모달 학습의 기본 목표는 여러 modality를 이용해 하나의 통합 표현을 학습하는 것이다.

```math
z = f\bigl(x^{(1)}, x^{(2)}, \dots, x^{(M)}\bigr)
```
각 항의 의미는 다음과 같다.
- $f(\cdot)$: multimodal encoder 또는 fusion 함수
- $z$: shared representation 또는 fused representation

최종 예측은 다음과 같이 쓸 수 있다.
```math
\hat{y} = g(z) = g\Bigl(f\bigl(x^{(1)}, x^{(2)}, \dots, x^{(M)}\bigr)\Bigr)
```

여기서,
- $g(\cdot)$: prediction head
- $\hat{y}$: 분류 결과, 회귀값, 생성 토큰 등 task별 출력

즉, 전체 구조는 대체로 두 단계로 이해할 수 있다.

- 각 modality에서 정보를 추출하고 통합 표현 $z$를 학습
- 그 표현을 이용해 downstream task 수행

### 1.4 Supervised Objective
지도학습에서는 데이터셋을 다음과 같이 둔다.
```math
\mathcal{D}=\{(x_i^{(1)},x_i^{(2)},\dots,x_i^{(M)},y_i)\}_{i=1}^{N}
```
여기서,
- $N$: 샘플 수
- $x_i^{(m)}$: $i$번째 샘플의 $m$번째 modality
- $y_i$: 정답 라벨 또는 타깃 값

일반적인 empirical risk minimization 목적함수는 다음과 같다.
```math
\min_{\theta}\frac{1}{N}\sum_{i=1}^{N}\mathcal{L}(\hat{y}_i,y_i)
```

여기서,
- $\theta$: 전체 모델 파라미터
- $\mathcal{L}$: 손실함수

분류 문제에서는 cross entropy를 자주 사용한다.
```math
\mathcal{L}_{CE} = -\sum_{c=1}^{C} y_{ic}\log \hat{y}_{ic}
```
- $C$: 클래스 수
- $y_{ic}$: 샘플 $i$가 클래스 $c$에 속하면 1, 아니면 0
- $\hat{y}_{ic}$: 클래스 $c$에 대한 예측 확률

회귀 문제에서는 mean squared error를 자주 사용한다.
```math
\mathcal{L}_{MSE} = (y_i-\hat{y}_i)^2
```
- $y_i$: 실제 타깃
- $\hat{y}_i$: 예측값
- 제곱을 쓰는 이유는 큰 오차에 더 큰 패널티를 주기 위해서다

### 1.5 Core Challenges
멀티모달 학습에서 반복적으로 등장하는 핵심 문제는 다음과 같다.
- 1) **Representation learning**: modality별 정보와 공통 의미를 반영하는 좋은 잠재표현을 학습하는 문제
- 2) **Alignment**: 서로 다른 modality에서 같은 의미를 갖는 샘플이 공통 공간에서 가깝게 위치하도록 맞추는 문제
- 3) **Fusion**: 여러 modality를 실제로 어떤 구조로 결합할 것인지 결정하는 문제
- 4) **Missing modality**: 학습 또는 추론 시점에 일부 modality가 빠져도 동작하게 만드는 문제
---
## Chapter 2. Classical Multimodal Methods
### 2.1 Early Fusion
가장 단순한 방법은 입력 단계에서 여러 modality를 하나의 벡터로 이어 붙이는 것이다.
```math
x = [x^{(1)};x^{(2)};\dots;x^{(M)}]
```
여기서 $[;]$는 concatenation 연산이다. 이후 모델은 이 결합 벡터를 입력으로 사용한다.
```math
z = f(x)
```

장점은 구조가 단순하고 구현이 쉽다는 점이다. 그러나 modality마다 분포와 의미 구조가 다르기 때문에, 단순 concatenation만으로는 modality 간 복잡한 상호작용을 잘 포착하지 못하는 경우가 많다.

### 2.2 Late Fusion

Late fusion은 각 modality를 독립적으로 처리한 뒤 마지막 단계에서 결합하는 방식이다.

```math
z^{(m)} = f_m(x^{(m)})
```

```math
\hat{y} = g(z^{(1)}, z^{(2)}, \dots, z^{(M)})
```

여기서,

- $f_m(\cdot)$: $m$번째 modality 전용 encoder
- $z^{(m)}$: 해당 modality의 표현
- $g(\cdot)$: 최종 결합 함수 또는 예측기

장점은 각 modality에 특화된 모델을 자유롭게 사용할 수 있다는 점이다. 반면, 각 modality가 독립적으로 처리되므로 cross-modal interaction을 충분히 학습하기 어렵다.

### 2.3 Intermediate or Hybrid Fusion

Intermediate fusion은 각 modality를 개별 encoder로 먼저 처리한 후 중간 representation 단계에서 결합하는 방식이다.

```math
z^{(1)} = f_1(x^{(1)}), \quad z^{(2)} = f_2(x^{(2)}), \dots, \quad z^{(M)} = f_M(x^{(M)})
```

```math
z = h(z^{(1)}, z^{(2)}, \dots, z^{(M)})
```

- $h(\cdot)$: fusion module
- $z$: 통합 표현

현대 multimodal 모델의 많은 구조가 사실상 이 범주에 들어간다. modality-specific encoder와 fusion module을 분리하면 유연성과 표현력을 동시에 확보할 수 있기 때문이다.

### 2.4 Canonical Correlation Analysis (CCA)

CCA는 고전적인 다중 뷰 학습 방법으로, 두 modality를 서로 최대한 상관되게 투영하는 선형 방법이다.

```math
\max_{w_1,w_2} \text{corr}(w_1^T x^{(1)}, w_2^T x^{(2)})
```

여기서,

- $w_1, w_2$: 각각 두 modality에 대한 projection vector
- $w_1^T x^{(1)}$, $w_2^T x^{(2)}$: 투영된 1차원 표현
- $\text{corr}(\cdot,\cdot)$: 상관계수

상관계수는 일반적으로 다음과 같이 정의된다.

```math
\text{corr}(a,b)=\frac{\text{cov}(a,b)}{\sqrt{\text{var}(a)\text{var}(b)}}
```

- $\text{cov}(a,b)$: 공분산
- $\text{var}(a)$, $\text{var}(b)$: 분산

CCA의 핵심은 두 modality가 공통으로 담고 있는 정보를 잘 보존하는 저차원 공간을 찾는 것이다. 다만 선형 투영만 허용되므로 복잡한 비선형 구조를 잘 담지 못한다.

### 2.5 Deep CCA (DCCA)

DCCA는 CCA를 신경망 기반 비선형 표현으로 확장한 방법이다.

```math
h_1 = f_1(x^{(1)}), \quad h_2 = f_2(x^{(2)})
```

```math
\max \text{corr}(h_1, h_2)
```

- $f_1, f_2$: 각 modality에 대한 neural encoder
- $h_1, h_2$: 비선형 latent representation

즉, 원래 CCA의 목적은 유지하되, raw input을 그대로 선형 투영하는 대신 먼저 깊은 표현을 학습한 뒤 그 표현들 간 상관관계를 최대화한다.

### 2.6 Limits of Classical Methods

전통적 방법들은 multimodal learning의 핵심 직관을 제공했지만 다음과 같은 한계를 가진다.

- 선형성 또는 얕은 구조로 인해 복잡한 의미적 상호작용 학습이 어려움
- 대규모 데이터와 고차원 입력에 대한 확장성이 제한적임
- 결측 modality나 잡음에 대한 강건성이 충분하지 않음
- 생성과 추론 중심의 현대적 multimodal task를 직접 다루기 어려움

이러한 한계가 이후 deep multimodal representation learning과 contrastive learning, Transformer 기반 모델로 이어지는 배경이 된다.

---

## Chapter 3. Deep Multimodal Representation Learning

### 3.1 Shared Embedding Space

딥러닝 기반 multimodal learning의 핵심 아이디어 중 하나는 서로 다른 modality를 공통 잠재공간에 매핑하는 것이다.

```math
z^{(1)} = f_1(x^{(1)}), \quad z^{(2)} = f_2(x^{(2)})
```

목표는 의미가 같은 샘플의 표현이 서로 가깝게 되도록 하는 것이다.

```math
z^{(1)} \approx z^{(2)}
```

여기서 $\approx$는 수학적 동일성이 아니라, embedding space에서 거리 또는 유사도가 작아지는 방향으로 학습된다는 의미이다. 이러한 shared embedding space는 retrieval, matching, cross-modal transfer의 기반이 된다.

### 3.2 Metric Learning Perspective

공통 공간에서 alignment를 학습하려면 어떤 거리 또는 유사도 개념이 필요하다. 자주 쓰이는 유사도는 cosine similarity이다.

```math
\text{sim}(u,v)=\frac{u^T v}{\|u\|\|v\|}
```

- $u^T v$: 두 벡터의 내적
- $\|u\|$, $\|v\|$: 각 벡터의 크기
- 값이 1에 가까울수록 방향이 유사함

cosine similarity는 embedding의 크기보다 방향 유사성에 집중하기 때문에 multimodal contrastive learning에서 널리 사용된다.

### 3.3 Contrastive Learning

현대 multimodal representation learning의 가장 중요한 축은 contrastive learning이다. 대표적인 loss는 다음과 같이 쓸 수 있다.

```math
\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i^{(a)}, z_i^{(b)})/\tau)}{\sum_j \exp(\text{sim}(z_i^{(a)}, z_j^{(b)})/\tau)}
```

각 항의 의미는 다음과 같다.

- $z_i^{(a)}$: anchor modality의 $i$번째 샘플 표현
- $z_i^{(b)}$: 같은 의미를 갖는 positive pair 표현
- $z_j^{(b)}$: 배치 안의 다른 negative sample 표현
- $\text{sim}(\cdot,\cdot)$: 보통 cosine similarity
- $\tau$: temperature parameter

이 식의 직관은 분명하다.

- 분자: 정답 쌍의 similarity를 크게 만들려 함
- 분모: 다른 모든 후보와 비교해 정답 쌍이 상대적으로 더 크도록 강제함

즉, 같은 의미의 multimodal pair는 가깝게, 다른 pair는 멀어지게 학습한다. 이 구조는 image-text alignment의 핵심 기반이 된다.

### 3.4 CLIP as a Turning Point

CLIP은 대규모 image-text pair에 대해 contrastive learning을 적용하여 vision-language 공동 표현을 학습한 대표적 모델이다. 구조는 크게 다음 두 부분으로 이루어진다.

- image encoder
- text encoder

각 encoder는 입력을 embedding space로 보낸다.

```math
z_i^{(v)} = f_v(x_i^{(v)}), \quad z_i^{(t)} = f_t(x_i^{(t)})
```

그 뒤 image-to-text, text-to-image 양방향 contrastive objective를 사용한다. CLIP의 중요성은 특정 task에 맞춘 supervision 없이도 대규모 natural language supervision만으로 transferable representation을 학습했다는 데 있다. 이후의 multimodal foundation model 흐름은 사실상 이 아이디어를 대규모화하고 생성 모델과 결합하는 방향으로 발전했다.

---

## Chapter 4. Multimodal Fusion, Attention, and Transformer

### 4.1 Attention Mechanism

Transformer와 현대 multimodal 모델의 핵심은 attention 메커니즘이다.

```math
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
```

각 항의 의미는 다음과 같다.

- $Q$: query matrix
- $K$: key matrix
- $V$: value matrix
- $QK^T$: query와 key의 유사도
- $\sqrt{d}$: dimension scaling term
- $\text{softmax}$: attention weight를 확률처럼 정규화
- 최종적으로 weighted sum of values를 계산

직관적으로 보면, query가 “무엇을 찾고 싶은지”를 나타내고, key는 “각 후보가 어떤 특성을 가지는지”를 나타내며, value는 “실제로 가져올 정보”를 나타낸다.

### 4.2 Self-Attention and Cross-Attention

Self-attention은 같은 modality 내부의 token들끼리 관계를 학습한다. 반면 cross-attention은 서로 다른 modality 사이의 관계를 학습한다.

예를 들어 text가 image를 참조하는 경우는 다음과 같이 볼 수 있다.

- $Q$: text token
- $K, V$: image patch token

이 경우 text query는 image representation 중 어떤 부분이 현재 문맥에 중요한지를 선택하게 된다. 따라서 cross-attention은 multimodal interaction을 직접 모델링하는 핵심 장치다.

### 4.3 Multimodal Transformer

Transformer 기반 multimodal 모델은 여러 modality token을 함께 입력받아 joint representation을 학습한다.

```math
Z = \text{Transformer}([X^{(1)}, X^{(2)}, \dots, X^{(M)}])
```

여기서,

- $X^{(m)}$: $m$번째 modality를 token sequence로 변환한 표현
- $[\,]$: token concatenation 또는 joint input formatting
- $Z$: multimodal contextualized representation

이 구조는 early fusion보다 유연하고, late fusion보다 상호작용을 잘 포착할 수 있다. 특히 긴 문맥과 복합 추론이 필요한 task에서 강력하다.

### 4.4 Tokenization Across Modalities

Transformer 기반 multimodal learning이 가능해진 이유 중 하나는 다양한 modality를 token sequence처럼 다룰 수 있게 되었기 때문이다.

- text: subword token
- image: patch token
- audio: frame token 또는 discrete code
- video: spatiotemporal token

즉, 서로 다른 modality라도 token sequence라는 공통 인터페이스로 변환하면, attention 기반의 통합 처리가 가능해진다.

---

## Chapter 5. Missing Modality and Robust Multimodal Learning

### 5.1 Why Missing Modality Matters

실제 환경에서는 모든 modality가 항상 완전하게 주어지지 않는다. 의료에서는 특정 검사 누락이 흔하고, 산업 환경에서는 센서 고장이나 통신 문제로 일부 signal이 빠질 수 있다. 자율주행에서도 특정 센서가 일시적으로 실패할 수 있다.

따라서 multimodal model은 다음 질문에 답해야 한다.

- 학습할 때는 여러 modality가 있었지만 추론할 때 일부가 없으면 어떻게 할 것인가
- 특정 modality 품질이 낮을 때 어떻게 강건하게 유지할 것인가

### 5.2 Modality Dropout

가장 직관적인 방법 중 하나는 학습 시 일부 modality를 랜덤하게 제거하는 modality dropout이다. 일반적인 dropout이 feature unit을 제거하듯이, modality dropout은 modality 단위를 제거해 결측 상황을 시뮬레이션한다. 이렇게 하면 모델이 특정 modality 하나에 과도하게 의존하는 것을 줄일 수 있다.

### 5.3 Generative Imputation

결측 modality를 생성 모델로 복원하는 방식도 있다. 예를 들어 text와 tabular 정보를 보고 missing image representation을 추정하거나, available modalities로부터 latent representation을 보간하는 방식이다. 이 접근은 강력하지만, 복원 오류가 downstream prediction에 전파될 수 있으므로 주의가 필요하다.

### 5.4 Shared Latent Representation

각 modality를 동일한 latent space로 매핑해 두면, 일부 modality가 없더라도 남은 modality만으로 유사한 latent representation을 구성하려는 전략을 사용할 수 있다. 이 접근은 retrieval, robustness, knowledge distillation과도 자연스럽게 연결된다.

---

## Chapter 6. Modern Multimodal Foundation Models

### 6.1 From Representation to Foundation Models

현대 multimodal learning은 단일 task 모델에서 foundation model로 이동했다. 중요한 흐름은 다음과 같다.

```text
representation learning → alignment → large-scale pretraining → instruction tuning → multimodal reasoning/generation
```

즉, 초기에는 공통 표현과 정렬이 핵심이었다면, 이제는 대규모 사전학습과 instruction tuning을 통해 다양한 task를 하나의 모델이 수행하게 되는 방향으로 발전했다.

### 6.2 CLIP

CLIP은 image-text contrastive learning을 통해 강력한 zero-shot vision-language representation을 학습했다. 핵심 기여는 자연어 supervision을 사용해 handcrafted label space에 덜 의존하는 transferable representation을 만든 것이다.

### 6.3 Flamingo

Flamingo는 강력한 language model과 vision encoder를 결합하여 few-shot multimodal prompting을 수행하는 구조로 알려졌다. 핵심 아이디어는 사전학습된 LLM을 유지한 채, visual token을 language model이 활용할 수 있도록 cross-attention 기반 adapter를 삽입하는 것이다.

### 6.4 BLIP and BLIP-2

BLIP 계열은 vision-language pretraining을 보다 통합적으로 수행하고, BLIP-2는 frozen image encoder와 frozen LLM 사이를 가볍고 효율적인 bridge module로 연결하는 구조를 제안했다. 이는 대규모 pretrained component를 재활용하면서도 강력한 multimodal generation을 가능하게 했다.

### 6.5 LLaVA and Instruction-Tuned Multimodal Models

LLaVA류 모델은 visual encoder와 LLM을 연결한 뒤 instruction tuning을 적용하여 multimodal chat 형태의 사용성을 크게 높였다. 여기서 중요한 점은 “이미지 이해” 자체뿐 아니라, “이미지를 보고 지시를 따르는 언어적 응답 생성”이 핵심 task로 부상했다는 점이다.

### 6.6 Broader Trend

최신 multimodal 모델의 흐름은 대체로 다음과 같이 요약할 수 있다.

- alignment 기반 representation learning
- pretrained unimodal model 재사용
- lightweight connector 또는 adapter
- instruction tuning
- reasoning과 generation의 통합

---

## Chapter 7. Applications of Multimodal Learning

### 7.1 Medical AI

의료는 대표적인 multimodal 분야이다. 가능한 입력 조합은 매우 다양하다.

- image + clinical text
- image + tabular biomarkers
- pathology + genomics
- longitudinal EMR + imaging

주요 task는 다음과 같다.

- diagnosis
- prognosis prediction
- treatment response prediction
- survival analysis
- clinical decision support

### 7.2 Industrial AI

산업 현장에서는 sensor, image, waveform, maintenance log가 함께 사용된다. 주요 task는 다음과 같다.

- anomaly detection
- fault diagnosis
- predictive maintenance
- root cause analysis

### 7.3 Multimodal Reasoning

기존에는 classification이나 retrieval이 중심이었지만, 최신 모델은 reasoning을 더 강조한다. 예를 들어 “이미지를 보고 추론하여 설명하기”, “문서와 표를 함께 읽고 답변하기” 같은 문제들이 여기에 해당한다.

### 7.4 Takeaways from Part 1

Multimodal learning의 핵심 흐름은 다음처럼 정리할 수 있다.

- classical fusion
- shared representation learning
- contrastive alignment
- attention-based interaction modeling
- foundation model and instruction tuning

즉, 이 분야는 단순한 feature concatenation에서 출발해, 이제는 reasoning-capable multimodal foundation model로 발전했다.

## Overall Roadmap and Final Summary
- classical fusion
- shared representation learning
- CCA와 DCCA
- contrastive alignment
- attention과 Transformer
- missing modality robustness
- foundation model로의 확장

# Part 2 (Advanced Extension). Multimodal Learning: Theory and Modern Practice
## Chapter 17. Information-Theoretic Foundations

멀티모달 학습은 서로 다른 modality가 공유하는 정보를 최대화하고, 불필요한 modality-specific 노이즈를 최소화하는 문제로 해석할 수 있다. 이를 정보이론적으로 표현하면 Mutual Information(MI) 최대화 문제로 볼 수 있다.

```math
I(X^{(1)}; X^{(2)}) = \mathbb{E}_{p(x^{(1)},x^{(2)})}\left[\log \frac{p(x^{(1)},x^{(2)})}{p(x^{(1)})p(x^{(2)})}\right]
```

- $p(x^{(1)},x^{(2)})$: 두 modality의 결합 분포  
- $p(x^{(1)})p(x^{(2)})$: 독립 가정 하의 분포  
- MI는 두 변수 간 공유 정보량  

representation 관점에서는 다음을 목표로 한다.

```math
\max I(z^{(1)}; z^{(2)})
```

즉, 서로 다른 modality의 latent representation이 공통 의미를 최대한 공유하도록 학습한다.

---

## Chapter 18. Contrastive Learning as MI Estimation

InfoNCE loss는 MI의 하한(lower bound)을 근사한다.

```math
\mathcal{L}_{InfoNCE} = - \mathbb{E}\left[\log \frac{\exp(\text{sim}(z_i^{(a)},z_i^{(b)})/\tau)}{\sum_{j}\exp(\text{sim}(z_i^{(a)},z_j^{(b)})/\tau)}\right]
```

- $z_i^{(a)}, z_i^{(b)}$: positive pair  
- $z_j^{(b)}$: negative samples  
- $\tau$: temperature  

이때 다음 관계가 성립한다.

```math
I(X;Y) \geq \log N - \mathcal{L}_{InfoNCE}
```

즉, loss를 최소화하는 것은 mutual information을 최대화하는 것과 연결된다. CLIP, ALIGN 등 대부분의 multimodal foundation model은 이 원리를 기반으로 한다.

---

## Chapter 19. Multimodal Objective Decomposition

실제 multimodal 학습은 단일 목적함수가 아니라 여러 목적의 결합이다.

```math
\min_{\theta} \mathcal{L}_{task} + \lambda_1 \mathcal{L}_{align} + \lambda_2 \mathcal{L}_{reg}
```

- $\mathcal{L}_{task}$: supervised loss (classification, regression)  
- $\mathcal{L}_{align}$: modality alignment (contrastive, CCA 등)  
- $\mathcal{L}_{reg}$: regularization 또는 consistency  

이 구조는 다음 의미를 가진다.

- task 성능을 유지하면서  
- modality 간 의미 정렬을 동시에 수행  

---

## Chapter 20. Modality Gap and Distribution Mismatch

멀티모달 학습에서 중요한 문제는 modality 간 분포 차이이다.

```math
d = \| z^{(1)} - z^{(2)} \|
```

이 거리 또는 분포 차이가 클수록 alignment가 어려워진다. 특히 다음 상황에서 문제가 심각해진다.

- 이미지 vs 텍스트 (semantic abstraction level 차이)
- 센서 vs 이미지 (noise 구조 차이)
- 임상 데이터 vs 영상 (scale 차이)

이를 해결하기 위한 접근은 다음과 같다.

- shared latent space learning  
- adversarial domain alignment  
- contrastive learning  
- distribution matching  

---

## Chapter 21. Optimal Transport for Multimodal Alignment

Optimal transport는 두 분포를 직접 정렬하는 방법이다.

```math
\min_{\gamma} \sum_{i,j} \gamma_{ij} c(x_i, y_j)
```

- $\gamma_{ij}$: transport plan  
- $c(x_i, y_j)$: cost function  

이 접근은 sample-level이 아닌 distribution-level alignment를 수행한다는 점에서 contrastive learning과 보완적인 관계를 가진다.

---

## Chapter 22. Multimodal Generalization and Domain Shift

멀티모달 환경에서도 domain shift는 중요한 문제다.

```math
P_{train}(x^{(m)}) \neq P_{test}(x^{(m)})
```

문제 유형:

- covariate shift  
- modality-specific shift  
- missing modality  

대표 해결 방법:

- domain invariant representation  
- IRM (Invariant Risk Minimization)  
- GroupDRO  

---

## Chapter 23. Causal Perspective in Multimodal Learning

최근 multimodal learning에서도 causal inference가 중요한 주제로 등장한다.

기존 모델:

```math
P(Y|X)
```

causal 모델:

```math
P(Y|do(X))
```

- $do(X)$: intervention  

멀티모달에서의 핵심 질문:

- 어떤 modality가 causal인가  
- 어떤 modality가 confounder인가  
- spurious correlation을 어떻게 제거할 것인가  

---


## Chapter 24. Probabilistic View of Language Models

LLM은 확률분포를 학습하는 모델이다.

```math
P(x_1,\dots,x_T)=\prod_{t=1}^{T}P(x_t|x_{1:t-1})
```

이는 chain rule을 적용한 결과이며, language modeling을 autoregressive prediction 문제로 바꾼다.

---

## Chapter 25. Token Distribution and Softmax

모델의 출력은 logits $z_i$로 표현되며, softmax를 통해 확률로 변환된다.

```math
P_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
```

temperature가 적용되면:

```math
P_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
```

- $T$가 작으면: sharp distribution  
- $T$가 크면: flat distribution  

---

## Chapter 26. Decoding Strategies

Beam search:

```math
\max \sum_{t} \log P(x_t|x_{1:t-1})
```

Sampling 기반 방법:

- top-k sampling  
- nucleus sampling  

trade-off:

- deterministic vs diversity  

---

## Chapter 27. Hallucination and Uncertainty

LLM은 확률 기반 모델이므로 사실이 아닌 출력도 생성할 수 있다.

```math
\hat{y} \sim P_\theta(y|x)
```

문제 원인:

- training distribution bias  
- knowledge limitation  
- over-generalization  

해결 접근:

- RAG  
- calibration  
- uncertainty modeling  

---

## Chapter 28. Alignment and RLHF

LLM alignment는 인간 선호를 반영하는 최적화 문제다.

```math
\max_{\theta} \mathbb{E}_{y \sim \pi_\theta}[r(y)]
```

- $r(y)$: reward model  

RLHF 단계:

- supervised fine-tuning  
- reward model 학습  
- policy optimization  

---

## Chapter 29. Scaling Law

```math
L(N) \propto N^{-\alpha}
```

- $N$: 데이터 크기  
- $\alpha$: scaling exponent  

의미:

- 데이터와 모델이 커질수록 성능이 power-law로 개선됨  

---

## Chapter 30. Multimodal LLM Integration
최신 AI 시스템은 multimodal과 LLM이 통합된 형태다.
```math
y = f(x^{text}, x^{image}, x^{audio}, \mathcal{K})
```
- $\mathcal{K}$: external knowledge  

구성 요소:
- perception (vision/audio encoder)  
- reasoning (LLM)  
- memory (retrieval system)  
---
## Chapter 31. Unified Perspective
전체 흐름은 다음과 같이 정리된다.

```text
Multimodal Representation → Alignment → Transformer → LLM → RLHF → RAG → Multimodal LLM
```

핵심 개념:

- representation learning  
- alignment  
- generation  
- reasoning  

---

# References
- Ngiam, J., Khosla, A., Kim, M., Nam, J., Lee, H., and Ng, A. Y. Multimodal Deep Learning. ICML, 2011.
- Srivastava, N., and Salakhutdinov, R. Multimodal Learning with Deep Boltzmann Machines. NeurIPS, 2012.
- Andrew, G., Arora, R., Bilmes, J., and Livescu, K. Deep Canonical Correlation Analysis. ICML, 2013.
- Baltrušaitis, T., Ahuja, C., and Morency, L.-P. Multimodal Machine Learning: A Survey and Taxonomy. IEEE TPAMI, 2019.
- Radford, A., Kim, J. W., Hallacy, C., et al. Learning Transferable Visual Models From Natural Language Supervision. ICML, 2021.
- Alayrac, J.-B., Donahue, J., Luc, P., et al. Flamingo: a Visual Language Model for Few-Shot Learning. NeurIPS, 2022.
- Li, J., Li, D., Savarese, S., and Hoi, S. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. ICML, 2023.
- Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual Instruction Tuning. NeurIPS, 2023.
- 
# Additional References
- Poole et al., "On Variational Bounds of Mutual Information", ICML 2019  
- Gutmann & Hyvärinen, "Noise Contrastive Estimation", 2010  
- Arjovsky et al., "Invariant Risk Minimization", 2019  
- Peyré & Cuturi, "Computational Optimal Transport", 2019  
- Kaplan et al., "Scaling Laws for Neural Language Models", 2020  
