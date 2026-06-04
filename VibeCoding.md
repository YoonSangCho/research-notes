# 바이브 코딩(Vibe Coding) 실습 가이드

- AI에게 자연어로 요청해서 코드를 만드는 개발 방식
- 코드 문법보다 **무엇을 만들지**에 집중하여 주문할 수 있어야 함
- **즉, 내가 사장이고 10000개 박사학위를 갖진 신입사원(LLM)이 들어온 상황**

---

## 목차

1. [실습 환경 1: VS Code + Claude](#실습-환경-1-vs-code--claude)
2. [실습 환경 2: Amazon Kiro](#실습-환경-2-amazon-kiro)
3. [두 환경 비교](#두-환경-비교)
4. [실습 프롬프트 예시: Adversarial Attacks & Defense](#실습-프롬프트-예시-adversarial-attacks--defense)

---

## 실습 환경 1: VS Code + Claude

### Visual Studio Code

Microsoft가 만든 무료 오픈소스 코드 에디터입니다. Windows, macOS, Linux를 모두 지원하며, 확장(Extension)을 통해 AI 코딩 도구와 쉽게 연동할 수 있습니다.

**다운로드:** https://code.visualstudio.com/download

### Claude란?

Anthropic이 만든 AI 어시스턴트입니다. 자연어로 코드를 요청하면 코드베이스 전체의 맥락을 이해하고 코드 생성, 수정, 디버깅을 도와줍니다.

### VS Code에 Claude 연동하기

**사전 준비**
- VS Code 1.98.0 이상
- Anthropic 계정 (claude.ai 가입, Pro 이상 플랜)

**설치 방법**

1. VS Code 실행 후 확장 탭 열기: `Ctrl+Shift+X` (Windows) / `Cmd+Shift+X` (Mac)
2. `Claude Code` 검색 → Anthropic 공식 확장 설치
3. 사이드바의 Claude 아이콘 클릭 → Sign in으로 로그인

또는 아래 주소를 VS Code Command Palette에 직접 입력:
```
vscode:extension/anthropic.claude-code
```

**주요 기능**
- 자연어로 기능 구현 요청
- 코드 변경 전 Diff 미리보기 후 적용
- 파일 @-멘션으로 특정 파일/라인 대화에 포함
- 대화 기록 유지 및 이어서 작업

---

## 실습 환경 2: Amazon Kiro

### Kiro란?

Amazon AWS가 만든 에이전트형 AI IDE로 VS Code와 Claude를 별도로 연동하는 대신, AI가 처음부터 IDE 안에 통합되어 있음

**다운로드:** https://kiro.dev/downloads/

Windows, macOS, Linux 모두 지원하며 현재 Preview 기간 중 무료로 사용할 수 있음

### 핵심 기능

**스펙 기반 개발 (Spec-Driven Development)**
자연어 요청을 받으면 바로 코드를 생성하는 대신, 요구사항 정의 → 시스템 설계 → 태스크 분리 → 코드 생성 순서로 진행하며, 복잡한 프로젝트에서 방향성을 잃지 않도록 도와줌

**에이전트 훅 (Agent Hooks)**
파일 저장, 커밋 등의 이벤트에 AI를 자동으로 트리거 함, 예를 들어 파일 저장 시 자동 린팅이나 보안 스캔을 실행할 수 있음

**스티어링 (Steering)**
팀의 코딩 컨벤션을 마크다운 파일로 정의하면 Kiro가 해당 규칙에 맞게 코드를 생성

**VS Code 호환**
기존 VS Code의 설정, 테마, 플러그인을 그대로 가져올 수 있음

### 설치 및 시작

1. https://kiro.dev/downloads/ 에서 OS에 맞는 파일 다운로드
2. 설치 실행 후 라이선스 동의
3. Google, GitHub, AWS Builder ID 중 하나로 로그인 (AWS 계정 불필요)
4. VS Code 설정 가져오기 선택 (선택사항)

---

## 두 환경 비교

| 항목 | VS Code + Claude | Amazon Kiro |
|------|-----------------|-------------|
| 설치 복잡도 | 중간 (별도 설정 필요) | 낮음 (올인원) |
| AI 통합 방식 | 확장 설치 방식 | 처음부터 내장 |
| 스펙 기반 개발 | 직접 구성 필요 | 기본 제공 |
| 에이전트 자동화 | 제한적 | Agent Hooks 제공 |
| VS Code 호환 | 그 자체가 VS Code | 설정 가져오기 지원 |
| 무료 사용 | Claude Pro 구독 필요 | Preview 중 무료 |
| 추천 대상 | 기존 VS Code 사용자 | 처음 시작하는 분 |

---

## 실습 프롬프트 예시: Adversarial Attacks & Defense

- Adversarial Attacks 강의에서 다룬 개념을 바이브 코딩으로 직접 실험해 봅니다.
- sklearn으로 tabular 데이터를 생성해서 실습하므로 노트북에서도 가볍게 실행됩니다.
- 아래 프롬프트를 Claude 또는 Kiro에 그대로 입력해서 코드를 생성해 보세요.

---

### 1. 실습용 tabular 데이터 생성

```
sklearn의 make_classification을 사용해서 이진 분류용 tabular 데이터를 생성해줘.
- 샘플 수: 2000개, 피처 수: 20개 (informative 10개, redundant 5개)
- 학습/테스트 8:2로 분리
- 피처 분포를 히스토그램으로 시각화하고, 클래스 비율도 출력해줘
- random_state=42로 고정해줘
```

---

### 2. FGSM 적대적 예제 생성 (tabular 버전)

```
위에서 만든 tabular 데이터에 FGSM을 적용하는 코드를 작성해줘.

- PyTorch로 간단한 MLP 분류 모델을 학습시켜
- FGSM으로 테스트 데이터에 perturbation을 추가해 적대적 예제를 생성해줘
- 엡실론 값은 0.05, 0.1, 0.2, 0.3으로 설정하고
- 각 엡실론별로 원본 데이터와 적대적 데이터의 피처값 변화량 분포를 나란히 시각화해줘
```

---

### 3. 엡실론별 모델 정확도 비교 실험

```
MLP 모델에 FGSM 공격을 가하면서 엡실론을 0.0부터 0.5까지 0.05 간격으로 변화시키는 실험을 설계해줘.

- 각 엡실론에서 적대적 예제를 생성하고 모델 정확도를 측정해
- x축: 엡실론, y축: 정확도로 결과를 그래프로 시각화해줘
- 엡실론 증가에 따른 취약성 변화를 확인할 수 있도록 해줘
```

---

### 4. Vanilla MLP vs Adversarial Training 모델 비교

```
위 실험에 방어 기법을 추가해줘.
일반 학습 모델(Vanilla MLP)과 Adversarial Training으로 학습한 모델(Robust MLP)을 비교하는 실험이야.

- 두 모델은 동일한 MLP 구조로 설계
- Robust MLP는 학습 중 FGSM(엡실론=0.1)으로 생성한 적대적 예제를 훈련 데이터에 혼합
- 엡실론 0.0~0.5 구간에서 두 모델의 정확도를 같은 그래프에 겹쳐서 시각화
- 결과 해석 주석도 코드에 포함해줘
```

---

### 5. 모델 아키텍처별 강건성 비교

```
동일한 tabular 데이터와 FGSM 실험을 아래 세 가지 모델에 대해 비교 실험하도록 확장해줘.
- Logistic Regression (sklearn)
- MLP (2 hidden layers, PyTorch)
- MLP (4 hidden layers, PyTorch)

엡실론 0.0~0.5 구간에서 세 모델의 정확도 변화를 한 그래프에 그리고,
결과를 pandas DataFrame으로 정리해서 CSV로 저장해줘.
```

---

> **팁:** 프롬프트가 너무 길면 단계별로 나눠서 요청하세요.
> "데이터 생성 코드만 먼저 작성해줘" → 실행 확인 → "이제 FGSM 루프 추가해줘" 순으로 진행하면 오류 없이 코드를 쌓아갈 수 있습니다.
