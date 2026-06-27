# Research Notes

머신러닝·딥러닝 주제별 연구 노트 모음. 각 노트는 **레퍼런스(표·모델 카탈로그·수식·실습 코드) + 교과서 수준 설명**을 함께 담는 것을 목표로 한다. 인용은 실존 문헌만 기재하며, 확정되지 않은 출처는 `[출처 확인 필요]`로 표기한다.

## 목차

| # | 노트 | 한 줄 요약 |
|---|---|---|
| 00 | [의료데이터 예측 모델링 프롬프트 템플릿](00.PredicvtiveModeling.md) | 데이터 유형별(정형·시계열·영상·텍스트·멀티모달) 모델 카탈로그 + 전처리·실험설계·평가·해석 치트시트 |
| 01 | [Multimodal Learning](01.MultimodalLearning.md) | 멀티모달 기초·융합·파운데이션 모델 + 정보이론·모달리티 갭·최적수송 등 심화 이론 |
| 02 | [Large Language Models](02.LargeLanguageModels.md) | 언어모델 기초, 트랜스포머, 스케일링, RLHF/DPO, RAG, PEFT(LoRA), 멀티모달 LLM |
| 03 | [Foundation Models](03.FoundationModels.md) | 파운데이션 모델의 정의와 데이터 타입별·산업별 지형도 |
| 04 | [Tabular Foundation Models](04.TabularFoundationModels.md) | 정형 데이터 딥러닝의 난점과 TabPFN 계열 in-context 학습 패러다임 |
| 05 | [Boosting](05.Boosting.md) | AdaBoost→GBM→XGBoost/LightGBM/CatBoost 발전사와 핵심 수식·비교 |
| 06 | [Federated Learning](06.FederatedLearning.md) | FedAvg/FedProx/SCAFFOLD 등 최적화, Non-IID, 프라이버시(DP·secure agg·gradient leakage) |
| 07 | [Survival Analysis](07.SurvivalAnaylsis.md) | 생존·위험함수, Cox PH, DeepSurv/DeepHit, C-index·IBS 평가 |
| 08 | [Fairness](08.Fairncess.md) | 공정성 지표(DP·EO·calibration), 불가능성 정리, pre/in/post-processing 기법 |
| 09 | [Domain Generalization](09.DomainGeneralization.md) | DG/DA/OOD 구분, 도메인 이동 유형, DANN·CORAL·IRM·GroupDRO, DomainBed |
| 10 | [Diffusion Models](10.DiffusionModels.md) | DDPM 수식, DDIM, score-based SDE, LDM/Stable Diffusion, consistency·flow matching |
| 11 | [BRCA Multimodal Foundation Models](11.BRCA_Multimodal_Foundation_Models.md) | 유방 MRI 기반 BRCA 변이 예측 — 라디오믹스에서 멀티모달 FM까지 실험 설계 *(작성 중)* |
| 12 | [Surrogate-Based Multi-Objective Optimization](12.SurrogateBasedMultiObjectiveOptimization.md) | 다목적 최적화·베이지안 최적화(EHVI·ParEGO)·제약 처리 — 다목적 공학 설계 최적화 사례 |
| 13 | [Surrogate Modeling & Computational Cost](13.SurrogateModeling_and_ComputationalCost.md) | 대리모델 선택과 계산비용, TabPFN의 최적화 루프 한계, conformal 불확실성 |
| 14 | [Causal Interpretation Pitfalls](14.CausalInterpretationPitfalls.md) | 상관≠인과 — 교란·Table 2 오류·역인과·collider, SHAP를 인과로 오해석하는 함정과 의료 사례 |
| 15 | [Prediction & Optimization Algorithms](15.PredictionAndOptimizationAlgorithms.md) | 다목적 공학 설계 최적화에 쓰는 예측모델 15종·최적화 6종을 연도순·항별 수식·실인용으로 정리한 배경지식 교과서 |
| 16 | [Statistical Testing](16.StatisticalTesting.md) | 의료데이터 통계검정 — 변수·가정별 검정 선택(t/Welch/Mann–Whitney/χ²/Fisher/Cochran–Armitage), 다중비교 보정, ROC·DeLong·보정·DCA, 내부/외부 검증과 TRIPOD |
| 17 | [Predictive Modeling Methods](17.PredictiveModelingMethods.md) | 예측모델 — 선형·정규화·SVM/kNN·트리·배깅/RF·부스팅·MLP/CNN/Transformer·파운데이션/전이·멀티모달 융합을 목적함수·학습절차·수식으로 비교 |
| 18 | [SHAP](18.SHAP.md) | 협력게임이론 Shapley 값·공리, SHAP 가법귀속, 주변 vs 조건부, Kernel/Tree/Deep/Linear 추정, 전역해석·상호작용, 의료 함정(인과 오독·상관·적대조작) |
| 19 | [Evaluating Prediction Models](19.EvaluatingPredictionModels.md) | 논문 "Statistical analysis" 단락 해부 — 부트스트랩 CI·사전지정 비교(선택편향)·DeLong·Bonferroni·보정(ECE/Brier)·DCA·macro-AUROC·민감도분석(ComBat/nested CV)·TRIPOD+AI/CLAIM |
| 20 | [Text Mining → Transformers → LLM](20.TextMiningToTransformersLLM.md) | 비전공자용 — 빈도(TF-IDF)→의미(word2vec)→문맥(트랜스포머 Q·K·V 색상·항별 풀이)→LLM, 토크나이저 종류(BPE/WordPiece/Unigram/SentencePiece), 어텐션의 변수선택 응용, 의료 NLP(BioBERT/Med-PaLM) |
| 21 | [Molecular Biology & Omics](21.MolecularBiologyAndOmics.md) | 중심원리(복제·전사·번역), 변이·후성유전, 시퀀싱·depth, 오믹스 계층·RNA-seq 정규화(TPM/CPM) — 유전체/단백질 FM 가교 |
| 22 | [Genomics & Precision Medicine](22.GenomicsAndPrecisionMedicine.md) | Hardy-Weinberg·GWAS·LD·유전율·PRS, 약물유전체·ACMG 변이해석 — 집단구조=교란을 인과 함정과 연결 |
| 23 | [Cell Signaling & Immunology](23.CellSignalingAndImmunology.md) | GPCR/RTK·MAPK·PI3K-AKT, 수용체 결합(K_d·Hill), 세포자멸사, 면역계·MHC·면역관문(PD-1/CTLA-4) |
| 24 | [Physiology & Biosignals](24.PhysiologyAndBiosignals.md) | 항상성·피드백, Nernst·Hodgkin-Huxley, 심박출량·ECG, GFR·청소율 — 생체신호(ECG/EEG/PPG)를 시계열로 연결 |
| 25 | [Pathology & Cancer Biology](25.PathologyAndCancerBiology.md) | 세포손상·염증·신생물(등급/병기), 종양유전자·억제유전자·Knudson 2-hit, Hallmarks of Cancer, Gompertzian 성장 |
| 26 | [Pharmacology & PK/PD](26.PharmacologyAndPKPD.md) | 용량반응 Emax/Hill·치료지수, ADME·반감기·청소율·분포용적·구획모델 — 용량 최적화를 대리모델 최적화와 연결 |
| — | [Vibe Coding](VibeCoding.md) | VS Code+Claude Code / Amazon Kiro 에이전트형 코딩 실습 가이드 |

> 파일명 일부에 오타가 있으나(`Predicvti`, `Fairncess`, `Anaylsis`) 기존 링크·git 이력 보존을 위해 유지한다.
