# Chapter 1. Multimodal Learning: 개요, 필요성, 문제 설정
## 1.1 Multimodal Learning의 필요성
인공지능이 현실 세계를 이해하기 위해서는 단일 데이터 형식만으로는 충분하지 않은 경우가 많다. 인간은 시각, 청각, 언어 등 다양한 정보를 동시에 활용하여 상황을 해석한다. 예를 들어, 이미지 정보는 공간적 구조를 제공하고, 텍스트는 의미를 제공하며, 음성은 감정과 억양을 전달한다.
의료 분야에서도 동일한 구조가 나타난다.

- MRI: 구조적 영상 정보
- EMR: 임상 기록 정보
- Lab test: 정량적 생리 정보
- Genomics: 분자 수준 정보

이처럼 현실 문제는 본질적으로 여러 데이터 표현을 포함하며, 이러한 서로 다른 데이터 표현을 **modality**라고 정의한다.

---

## 1.2 Multimodal Learning 정의

Multimodal Learning은 둘 이상의 modality를 동시에 활용하여 더 풍부하고 강건한 표현을 학습하는 방법이다.

대표적인 응용은 다음과 같다.

- 이미지 + 텍스트: image captioning, visual question answering
- 음성 + 영상: audio-visual speech recognition
- 의료: MRI + clinical + pathology
- 산업: sensor + image + log

핵심은 단순한 결합이 아니라, 서로 다른 modality가 제공하는 상보적 정보를 효과적으로 통합하는 것이다.

---

## 1.3 수학적 문제 정의

멀티모달 입력이 총 $M$개 존재한다고 가정한다.

```math
x^{(1)}, x^{(2)}, \dots, x^{(M)}
```

- $M$: modality 개수
- $x^{(m)}$: $m$번째 modality

---

### 표현 학습

```math
z = f(x^{(1)}, x^{(2)}, \dots, x^{(M)})
```

- $f$: multimodal fusion 함수
- $z$: shared representation

---

### 예측

```math
\hat{y} = g(z)
```

```math
\hat{y} = g(f(x^{(1)}, x^{(2)}, \dots, x^{(M)}))
```

---

## 1.4 주요 문제

멀티모달 학습의 핵심 문제는 다음 네 가지이다.

- Representation Learning
- Alignment
- Fusion
- Missing Modality

---

## 1.5 지도학습 목적함수

```math
\mathcal{D} = \{(x_i^{(1)}, ..., x_i^{(M)}, y_i)\}_{i=1}^N
```

```math
\min_{\theta} \frac{1}{N}\sum \mathcal{L}(\hat{y}_i, y_i)
```

# Chapter 2. Classical Multimodal Methods

---

## 2.1 Early Fusion

```math
x = [x^{(1)} ; x^{(2)}]
```

- 입력 단계 결합
- 단순하지만 표현력 부족

---

## 2.2 Late Fusion

```math
z^{(1)} = f_1(x^{(1)})
z^{(2)} = f_2(x^{(2)})
```

```math
\hat{y} = g(z^{(1)}, z^{(2)})
```

- modality 독립 학습
- interaction 부족

---

## 2.3 CCA

```math
\max \text{corr}(w_1^T x^{(1)}, w_2^T x^{(2)})
```

- 두 modality 정렬

---

## 2.4 DCCA

```math
h_1 = f_1(x^{(1)}), h_2 = f_2(x^{(2)})
```

```math
\max \text{corr}(h_1, h_2)
```

---

## 2.5 한계

- interaction 부족
- 표현력 부족
- scalability 문제

## 1.6 Chapter Summary

- Multimodal = 여러 데이터 표현 통합 문제
- 핵심 = alignment + fusion + representation
- 이후 deep multimodal로 확장됨

# Chapter 3. Deep Multimodal Representation Learning

---

## 3.1 Joint Embedding

```math
z^{(1)} = f_1(x^{(1)})
z^{(2)} = f_2(x^{(2)})
```

```math
z^{(1)} \approx z^{(2)}
```

---

## 3.2 Contrastive Learning

```math
\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i^{(v)}, z_i^{(t)})/\tau)}{\sum_j \exp(\text{sim}(z_i^{(v)}, z_j^{(t)})/\tau)}
```

- numerator: positive pair
- denominator: negative pairs

---

## 3.3 의미

- alignment 수행
- shared latent space 형성

---

## 3.4 CLIP 구조

- image encoder
- text encoder
- contrastive objective

# Chapter 4. Multimodal Fusion & Attention

---

## 4.1 Attention

```math
\text{Attention}(Q,K,V) = \text{softmax}(QK^T / \sqrt{d}) V
```

---

## 4.2 Cross Attention

- Q: text
- K,V: image

→ 텍스트가 이미지의 어떤 부분을 참조하는지 학습

---

## 4.3 Multimodal Transformer

```math
Z = \text{Transformer}([X^{(v)}, X^{(t)}])
```

---

## 4.4 특징

- modality interaction 학습 가능
- deep fusion 가능


# Chapter 5. Missing Modality

---

## 5.1 문제

- 일부 modality 없음
- real-world에서 필수 문제

---

## 5.2 해결 방법

- modality dropout
- imputation
- shared representation

---

## 5.3 핵심

- robustness 확보
- generalization 향상

- # Chapter 6. Modern Multimodal Models

---

## 6.1 CLIP

- contrastive learning
- image-text alignment

---

## 6.2 Flamingo

- frozen LLM + vision encoder

---

## 6.3 BLIP / BLIP-2

- vision-language pretraining

---

## 6.4 LLaVA

- multimodal chat model

---

## 6.5 핵심 흐름

- representation → alignment → generation

# Chapter 7. Applications

---

## 7.1 Medical AI

- MRI + EMR
- prognosis prediction

---

## 7.2 Industrial AI

- sensor + image
- anomaly detection

---

## 7.3 Foundation Model

- general multimodal reasoning
