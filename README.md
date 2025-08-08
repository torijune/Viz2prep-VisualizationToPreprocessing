# Viz2prep: Visualization to Preprocessing

전문화된 다중 에이전트 시스템을 사용하여 시각화 기반 데이터 전처리를 수행하는 프로젝트입니다.

## 🎯 프로젝트 목표

LangGraph를 활용한 전문화된 다중 에이전트 시스템으로 체계적이고 자동화된 데이터 전처리 워크플로우를 구현합니다.

## 🏗️ 새로운 전문화된 다중 에이전트 아키텍처

### 📊 전체 아키텍처
```
EDA Layer (5개 병렬) → Planning Layer (5개 순차) → 
Coding Layer (5개 순차) → Execution Layer (1개) → Response Layer (1개)
```

### 🔍 구성 요소

#### 1. EDA Layer (5개 전문 에이전트)
- **`numeric_agent`**: 수치형 변수 전문 분석
- **`category_agent`**: 범주형 변수 전문 분석  
- **`outlier_agent`**: 이상치 전문 분석
- **`nulldata_agent`**: 결측값 전문 분석
- **`corr_agent`**: 상관관계 전문 분석

#### 2. Planning Layer (5개 전문 플래너)
- **`numeric_planner`**: 수치형 전처리 계획 수립
- **`category_planner`**: 범주형 전처리 계획 수립
- **`outlier_planner`**: 이상치 전처리 계획 수립
- **`nulldata_planner`**: 결측값 전처리 계획 수립
- **`corr_planner`**: 상관관계 전처리 계획 수립

#### 3. Coding Layer (5개 전문 코더)
- **`numeric_coder`**: Knowledge Base 활용 수치형 코드 생성
- **`category_coder`**: Knowledge Base 활용 범주형 코드 생성
- **`outlier_coder`**: Knowledge Base 활용 이상치 코드 생성
- **`nulldata_coder`**: Knowledge Base 활용 결측값 코드 생성
- **`corr_coder`**: Knowledge Base 활용 상관관계 코드 생성

#### 4. Execution & Debug & Response Layer (3개)
- **`executor`**: 5개 도메인 코드를 순서대로 통합 실행
- **`debug_agent`**: 코드 오류 자동 디버깅 및 수정
- **`responder`**: LLM 기반 최종 응답 생성

### 🗺️ 워크플로우 State 구조

#### 📊 INPUT STATES (2개)
- `query`: 사용자의 전처리 요청
- `dataframe`: 원본 데이터프레임

### 🔄 워크플로우 플로우

```
START → EDA Layer (5개 병렬) → Planning Layer (5개 순차) → 
Coding Layer (5개 순차) → Execution Layer → Debug Layer → 
조건부 라우팅: Debug 성공 시 → Execution Layer (재실행) → Response Layer
Debug 실패 시 → Response Layer
```

#### 🔍 EDA STATES (5개)
- `numeric_eda_result`: 수치형 변수 EDA 결과
- `category_eda_result`: 범주형 변수 EDA 결과
- `outlier_eda_result`: 이상치 EDA 결과
- `nulldata_eda_result`: 결측값 EDA 결과
- `corr_eda_result`: 상관관계 EDA 결과

#### 📋 PLANNING STATES (5개)
- `numeric_plan`: 수치형 전처리 계획
- `category_plan`: 범주형 전처리 계획
- `outlier_plan`: 이상치 전처리 계획
- `nulldata_plan`: 결측값 전처리 계획
- `corr_plan`: 상관관계 전처리 계획

#### 💻 CODING STATES (5개)
- `numeric_code`: 수치형 전처리 코드
- `category_code`: 범주형 전처리 코드
- `outlier_code`: 이상치 전처리 코드
- `nulldata_code`: 결측값 전처리 코드
- `corr_code`: 상관관계 전처리 코드

#### 🔧 EXECUTION STATES (3개)
- `processed_dataframe`: 최종 전처리된 데이터프레임
- `execution_results`: 각 전처리 단계별 실행 결과
- `execution_errors`: 실행 중 발생한 오류들

#### 🐛 DEBUG STATES (4개)
- `debug_status`: 디버깅 상태
- `debug_message`: 디버깅 메시지
- `fixed_preprocessing_code`: 수정된 전처리 코드
- `debug_analysis`: 디버깅 분석 결과

#### ✅ OUTPUT STATES (2개)
- `final_answer`: 최종 응답
- `preprocessing_summary`: 전처리 요약 보고서

## 🚀 설치 및 실행

### 기본 설치
```bash
pip install -r requirements.txt
```

### 환경 변수 설정
`.env` 파일을 생성하고 OpenAI API 키를 설정하세요:

```
OPENAI_API_KEY=your_api_key_here
```

### 새로운 워크플로우 실행
```bash
python new_workflow_graph.py
```

### 디버깅 기능 테스트
```bash
python test_debug_workflow.py
```

## 🐛 디버깅 기능

### 🔧 Code Debug Agent
전처리 코드 실행 시 발생하는 오류를 자동으로 분석하고 수정하는 전문 에이전트입니다.

#### 주요 기능
- **오류 패턴 인식**: target_variable 누락, 컬럼 누락, 데이터 타입 오류 등 일반적인 오류 패턴을 자동으로 인식
- **지능적 코드 수정**: 오류 유형에 따라 적절한 수정 전략을 적용
- **안전한 재실행**: 수정된 코드로 자동 재실행하여 전처리 완료 보장

#### 지원하는 오류 유형
1. **target_variable 누락**: 사용 가능한 컬럼 중에서 적절한 target_variable 선택 또는 생성
2. **컬럼 누락**: 유사한 컬럼 찾기 또는 기본값으로 컬럼 생성
3. **데이터 타입 오류**: 안전한 데이터 타입 변환 적용
4. **NaN 처리 오류**: 범주형/수치형에 따른 적절한 결측값 처리
5. **일반적인 오류**: try-catch 블록으로 안전한 실행 환경 제공

#### 워크플로우 통합
- Execution Layer에서 오류 발생 시 자동으로 Debug Agent로 전달
- 디버깅 성공 시 수정된 코드로 재실행
- 디버깅 실패 시 오류 정보와 함께 최종 응답 생성

## 📁 프로젝트 구조

```
Viz2prep-VisualizationToPreprocessing/
├── new_workflow_graph.py          # 메인 워크플로우 실행 파일
├── workflow_state.py              # 워크플로우 State 정의
├── agents/                        # 전문화된 에이전트들
│   ├── eda_nodes.py              # EDA 노드들 (5개)
│   ├── planning_nodes.py         # Planning 노드들 (5개)
│   ├── coding_nodes.py           # Coding 노드들 (5개)
│   ├── execution_response_nodes.py # Execution & Response 노드들 (2개)
│   ├── eda_agents/               # 기존 EDA 에이전트들 (참조용)
│   └── preprocessing_agents/     # 기존 전처리 에이전트들 (참조용)
├── KB_rag_agents/                # Knowledge Base RAG 시스템
│   ├── KB_rag_agent.py          # RAG Agent
│   └── knowledge_base/          # Knowledge Base
├── generated_plots/              # 생성된 시각화 파일들
├── datasets/                     # 데이터셋
└── requirements.txt              # 의존성 패키지
```

## 🎯 주요 특징

### 1. 디버깅 친화적 설계
- **상세한 주석**: 모든 노드에 입출력 state와 다음 엣지 명시
- **실행 추적**: 각 단계별 상태 변화와 결과 출력
- **오류 처리**: 실패한 단계도 추적하여 디버깅 정보 제공

### 2. 모듈화 및 확장성
- **도메인별 전문화**: 각 에이전트가 특정 영역에 집중
- **Knowledge Base 활용**: 검증된 전처리 기법들 자동 검색
- **유연한 그래프 구조**: 새로운 노드 추가 용이

### 3. 상태 관리
- **22개 State 체계**: 전체 워크플로우 상태를 명확히 정의
- **노드별 역할 분담**: 각 노드가 담당하는 state 명확히 구분
- **엣지 연결성**: 어떤 state가 어느 노드로 전달되는지 추적 가능

## 📈 성과 지표

- **✅ 노드 수**: 17개 (EDA 5개 + Planning 5개 + Coding 5개 + Execution 1개 + Response 1개)
- **✅ State 수**: 22개 (체계적 상태 관리)
- **✅ 전문화 도메인**: 5개 (수치형, 범주형, 이상치, 결측값, 상관관계)
- **✅ 디버깅 주석**: 각 노드별 상세 입출력 및 엣지 정보
- **✅ Knowledge Base 연동**: 모든 코더 노드에서 KB 활용

## 🔄 워크플로우 실행 예시

```python
from new_workflow_graph import build_specialized_workflow

# 워크플로우 구축
workflow = build_specialized_workflow()

# 초기 상태 설정
initial_state = {
    "query": "이 데이터를 머신러닝 모델 학습에 적합하도록 전처리해주세요.",
    "dataframe": your_dataframe
}

# 워크플로우 실행
result = workflow.invoke(initial_state)

# 결과 확인
print(result["final_answer"])
print(result["preprocessing_summary"])
```

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 