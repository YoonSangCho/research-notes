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

---

## 1.6 Chapter Summary

- Multimodal = 여러 데이터 표현 통합 문제
- 핵심 = alignment + fusion + representation
- 이후 deep multimodal로 확장됨
