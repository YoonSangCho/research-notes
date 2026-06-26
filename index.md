---
hide:
  - navigation
  - toc
---

# Research Notes

머신러닝·딥러닝 주제별 **공부 노트 모음**입니다. 각 노트는 *레퍼런스(표·모델 카탈로그·수식·실습 코드)* 와 *교과서 수준 설명* 을 함께 담는 것을 목표로 합니다. 모든 인용은 실존 문헌이며, 확정되지 않은 출처는 표기해 둡니다.

아래 카드를 누르면 바로 해당 노트로 이동합니다. 상단의 검색으로 전체 노트를 한 번에 찾을 수 있고, 우측 상단 토글로 다크 모드를 켤 수 있습니다.

## 1. 예측모델

<div class="grid cards" markdown>

-   :material-clipboard-text-outline:{ .lg .middle } &nbsp;**[(1) 예측 모델링 템플릿](00.PredicvtiveModeling.md)**

    ---

    데이터 유형별(정형·시계열·영상·텍스트·멀티모달) 모델 카탈로그 + 전처리·실험설계·평가·해석 치트시트

-   :material-sitemap-outline:{ .lg .middle } &nbsp;**[(2) 예측 모델 방법론](17.PredictiveModelingMethods.md)**

    ---

    선형·정규화·SVM/kNN·트리·배깅/RF·부스팅·MLP/CNN/Transformer·파운데이션·멀티모달 융합을 목적함수·수식으로 비교

-   :material-check-decagram-outline:{ .lg .middle } &nbsp;**[(3) 예측 모델 평가](19.EvaluatingPredictionModels.md)**

    ---

    부트스트랩 CI·사전지정 비교(선택편향)·DeLong·Bonferroni·보정(ECE/Brier)·DCA·macro-AUROC·TRIPOD+AI/CLAIM

-   :material-chart-line:{ .lg .middle } &nbsp;**[(4) Boosting](05.Boosting.md)**

    ---

    AdaBoost → GBM → XGBoost / LightGBM / CatBoost 발전사와 핵심 수식·비교

-   :material-function-variant:{ .lg .middle } &nbsp;**[(5) 통계 검정](16.StatisticalTesting.md)**

    ---

    변수·가정별 검정 선택(t/Welch/Mann–Whitney/χ²/Fisher), 다중비교 보정, ROC·DeLong·보정·DCA, 내부/외부 검증과 TRIPOD

-   :material-lightbulb-on-outline:{ .lg .middle } &nbsp;**[(6) SHAP](18.SHAP.md)**

    ---

    협력게임이론 Shapley 값·공리, 가법귀속, 주변 vs 조건부, Kernel/Tree/Deep/Linear 추정, 의료 해석 함정

-   :material-alert-octagon-outline:{ .lg .middle } &nbsp;**[(7) 인과 해석의 함정](14.CausalInterpretationPitfalls.md)**

    ---

    상관 ≠ 인과 — 교란·Table 2 오류·역인과·collider, SHAP를 인과로 오해석하는 함정과 의료 사례

</div>

## 2. 딥러닝 및 멀티모달

<div class="grid cards" markdown>

-   :material-vector-combine:{ .lg .middle } &nbsp;**[(1) Multimodal Learning](01.MultimodalLearning.md)**

    ---

    멀티모달 기초·융합·파운데이션 모델 + 정보이론·모달리티 갭·최적수송 등 심화 이론

-   :material-message-text-outline:{ .lg .middle } &nbsp;**[(2) Large Language Models](02.LargeLanguageModels.md)**

    ---

    언어모델 기초, 트랜스포머, 스케일링, RLHF/DPO, RAG, PEFT(LoRA), 멀티모달 LLM

-   :material-book-open-page-variant-outline:{ .lg .middle } &nbsp;**[(3) Text Mining → Transformers → LLM](20.TextMiningToTransformersLLM.md)**

    ---

    비전공자용 — 빈도(TF-IDF) → 의미(word2vec) → 문맥(어텐션 Q·K·V 항별 풀이) → LLM, 토크나이저, 의료 NLP

-   :material-earth:{ .lg .middle } &nbsp;**[(4) Foundation Models](03.FoundationModels.md)**

    ---

    파운데이션 모델의 정의와 데이터 타입별·산업별 지형도

-   :material-table-large:{ .lg .middle } &nbsp;**[(5) Tabular Foundation Models](04.TabularFoundationModels.md)**

    ---

    정형 데이터 딥러닝의 난점과 TabPFN 계열 in-context 학습 패러다임

-   :material-blur:{ .lg .middle } &nbsp;**[(6) Diffusion Models](10.DiffusionModels.md)**

    ---

    DDPM 수식, DDIM, score-based SDE, LDM/Stable Diffusion, consistency·flow matching

</div>

## 3. 인공지능 신뢰성

<div class="grid cards" markdown>

-   :material-lan:{ .lg .middle } &nbsp;**[(1) Federated Learning](06.FederatedLearning.md)**

    ---

    FedAvg/FedProx/SCAFFOLD 등 최적화, Non-IID, 프라이버시(DP·secure agg·gradient leakage)

-   :material-transit-connection-variant:{ .lg .middle } &nbsp;**[(2) Domain Generalization](09.DomainGeneralization.md)**

    ---

    DG/DA/OOD 구분, 도메인 이동 유형, DANN·CORAL·IRM·GroupDRO, DomainBed

-   :material-scale-balance:{ .lg .middle } &nbsp;**[(3) Fairness](08.Fairncess.md)**

    ---

    공정성 지표(DP·EO·calibration), 불가능성 정리, pre/in/post-processing 완화 기법

-   :material-heart-pulse:{ .lg .middle } &nbsp;**[(4) Survival Analysis](07.SurvivalAnaylsis.md)**

    ---

    생존·위험함수, Cox PH, DeepSurv/DeepHit, C-index·IBS 평가

</div>

## 4. 최적화 및 대리모델

<div class="grid cards" markdown>

-   :material-target:{ .lg .middle } &nbsp;**[(1) Surrogate-Based Multi-Objective Optimization](12.SurrogateBasedMultiObjectiveOptimization.md)**

    ---

    파레토 지배·하이퍼볼륨, 베이지안 최적화(qEHVI·qNEHVI·qParEGO), 제약 처리 — 다목적 공학 설계 예시

-   :material-speedometer:{ .lg .middle } &nbsp;**[(2) Surrogate Modeling & Computational Cost](13.SurrogateModeling_and_ComputationalCost.md)**

    ---

    대리모델 선택과 계산비용 트레이드오프, TabPFN의 최적화 루프 한계, conformal 불확실성

-   :material-format-list-numbered:{ .lg .middle } &nbsp;**[(3) Prediction & Optimization Algorithms](15.PredictionAndOptimizationAlgorithms.md)**

    ---

    예측모델 15종·최적화 6종을 연도순·항별 수식·실인용으로 정리한 배경지식 교과서

</div>

## 5. 산업응용

<div class="grid cards" markdown>

-   :material-image-multiple-outline:{ .lg .middle } &nbsp;**[(1) BRCA Multimodal Foundation Models](11.BRCA_Multimodal_Foundation_Models.md)** &nbsp;*(작성 중)*

    ---

    유방 MRI 기반 BRCA 변이 예측을 예시로, 라디오믹스부터 멀티모달 파운데이션 모델까지 비교하는 실험 설계

-   :material-laptop:{ .lg .middle } &nbsp;**[(2) Vibe Coding](VibeCoding.md)**

    ---

    VS Code + Claude Code / Amazon Kiro 등 에이전트형 코딩 실습 가이드

</div>

---

!!! note "이 사이트에 대하여"
    GitHub 저장소의 마크다운 노트를 그대로 읽어 자동 발행합니다. 노트(`.md`)를 수정·추가하면 **GitHub Actions가 자동으로 사이트를 다시 빌드**합니다. 소스: [GitHub 저장소](https://github.com/YoonSangCho/research-notes)
