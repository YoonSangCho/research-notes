# 1. Multimodal Learning

## 1.1 정의

멀티모달 학습(multimodal learning)은 서로 다른 데이터 형식(modality), 예를 들어 이미지, 텍스트, 음성, 시계열, 표 데이터, 유전체 데이터를 함께 이용하여 더 나은 표현과 예측을 학습하는 패러다임이다.

하나의 샘플은 다음과 같이 표현된다:

$$
x = \{x^{(1)}, x^{(2)}, ..., x^{(M)}\}
$$

여기서

- $M$: modality 개수
- $x^{(m)}$: m번째 modality 입력

예시:

- $x^{(1)}$: MRI 영상  
- $x^{(2)}$: EMR  
- $x^{(3)}$: ECG  

목표는 다음과 같다:

$$
y = F(x^{(1)}, x^{(2)}, ..., x^{(M)})
$$

---

## 1.2 멀티모달이 필요한 이유

각 modality는 서로 다른 정보를 담고 있다:

- 영상 → 공간 구조  
- 텍스트 → 의미 정보  
- 시계열 → 동적 변화  
- 표 데이터 → 상태 요약  

따라서 멀티모달은 다음을 달성한다:

1. 정보 보완성 (complementarity)
2. 표현 정렬 (alignment)

---

## 1.3 Representation Learning

각 modality는 encoder를 통해 latent space로 매핑된다:

$$
z^{(m)} = f_m(x^{(m)})
$$

### 항 설명

- $f_m$: modality encoder  
- $z^{(m)}$: latent representation  

---

## 1.4 Fusion

### (1) Concatenation

$$
z = [z^{(1)}; z^{(2)}; ...; z^{(M)}]
$$

### (2) Weighted Fusion

$$
z = \sum_{m=1}^{M} w_m z^{(m)}
$$

---

## 1.5 Attention 기반 Fusion

$$
z = \sum_{m=1}^{M} \alpha_m z^{(m)}
$$

$$
\alpha_m = \frac{\exp(e_m)}{\sum_{k=1}^{M} \exp(e_k)}
$$

### 의미

- $e_m$: 중요도 score  
- $\alpha_m$: attention weight  

---

## 1.6 Contrastive Learning (CLIP)

$$
L = -\log \frac{\exp(sim(z_i^I, z_i^T)/\tau)}{\sum_{j=1}^{N} \exp(sim(z_i^I, z_j^T)/\tau)}
$$

### 항 설명

- $sim$: cosine similarity  
- $\tau$: temperature  
- numerator: positive pair  
- denominator: negative 포함  

---

## 1.7 Cross-Attention

$$
Attention(Q,K,V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

---

## 1.8 Multimodal 의료 모델

$$
\hat{y} = h(g(f_{img}(x_{img}), f_{ehr}(x_{ehr}), f_{sig}(x_{sig})))
$$

---

## 1.9 연구 흐름

- Early fusion → 단순 결합  
- CLIP → alignment  
- Flamingo → cross-attention  
- LLaVA → multimodal LLM  

---

## 1.10 핵심 문제

- modality gap  
- missing modality  
- domain shift  
- interpretability  
