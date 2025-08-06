# Viz2prep: Visualization to Preprocessing

멀티모달 LLM 에이전트를 사용하여 시각화 기반 데이터 전처리 성능을 평가하는 프로젝트입니다.

## 프로젝트 목표

통계 요약과 시각화 플롯을 모두 제공받은 멀티모달 LLM이 유용한 전처리 코드를 생성할 수 있는지 평가합니다.

## 새로운 RAG 기반 아키텍처

### 개요
EDA 결과 → Planning Agent → KB RAG Agent → 코드 생성

### 구성 요소

#### 1. Planning Agent (`planning_agents/`)
- **역할**: EDA 결과를 분석하여 전처리 작업 계획 수립
- **기능**:
  - EDA 결과 분석 및 전처리 필요성 판단
  - 작업 우선순위 결정
  - 의존성 검사 및 순서 조정
  - LLM 기반 계획 생성

#### 2. KB RAG Agent (`KB_rag_agents/`)
- **역할**: Knowledge Base를 활용한 RAG 기반 코드 생성
- **기능**:
  - 전처리 코드 Knowledge Base 구축
  - OpenAI Embedding을 사용한 유사도 검색
  - 관련 코드 예제 검색 및 참조
  - LLM 기반 코드 생성

#### 3. Knowledge Base 구조
```
KB_rag_agents/
├── knowledge_base/
│   ├── preprocessing_codes.json  # 전처리 코드 예제
│   └── embeddings.pkl           # 임베딩 벡터
├── KB_rag_agent.py             # RAG Agent
└── manage_kb.py                # Knowledge Base 관리 도구
```

### 전처리 카테고리
1. **Missing Values** (결측값 처리)
2. **Outliers** (이상치 처리)
3. **Categorical Encoding** (범주형 인코딩)
4. **Scaling** (스케일링)
5. **Feature Selection** (특성 선택)
6. **Feature Engineering** (특성 엔지니어링)
7. **Dimensionality Reduction** (차원 축소)
8. **Class Imbalance** (클래스 불균형)

## 기존 아키텍처

1. **DataLoader**: CSV 데이터를 로드
2. **StatisticsAgent**: 데이터 통계 요약 생성
3. **VisualizationAgent**: EDA 플롯 생성
4. **PreprocessingAgent**: 멀티모달 LLM을 사용한 전처리 코드 생성
5. **Evaluator**: ML 성능 비교 평가

## 설치 및 실행

### 기본 설치
```bash
pip install -r requirements.txt
```

### 환경 변수 설정
`.env` 파일을 생성하고 OpenAI API 키를 설정하세요:

```
OPENAI_API_KEY=your_api_key_here
```

### 테스트 실행

#### RAG 기반 워크플로우 테스트
```bash
python test_rag_workflow.py
```

#### 기존 시스템 테스트
```bash
python main.py
```

## Knowledge Base 관리

### Knowledge Base 구조
Knowledge Base는 JSON 파일로 관리되며, 각 전처리 기법은 다음 구조를 가집니다:

```json
{
  "category_name": {
    "description": "카테고리 설명",
    "techniques": [
      {
        "name": "기법 이름",
        "code": "실행 가능한 Python 코드",
        "description": "기법 설명",
        "use_case": "사용 사례",
        "keywords": ["키워드1", "키워드2", ...]
      }
    ]
  }
}
```

### Knowledge Base 관리 도구

#### 관리 도구 실행
```bash
python KB_rag_agents/manage_kb.py
```

#### 주요 기능
1. **카테고리 목록 보기**: 현재 Knowledge Base의 모든 카테고리 확인
2. **기법 목록 보기**: 특정 카테고리의 모든 기법 확인
3. **기법 검색**: 키워드 기반으로 관련 기법 검색
4. **기법 추가**: 새로운 전처리 기법 추가
5. **기법 제거**: 기존 기법 제거
6. **기법 업데이트**: 기존 기법 수정
7. **임베딩 재생성**: Knowledge Base 변경 후 임베딩 재생성
8. **Knowledge Base 검증**: 구조 검증

#### 새로운 기법 추가 예시
```bash
# 관리 도구 실행
python KB_rag_agents/manage_kb.py

# 메뉴에서 4번 선택 (기법 추가)
# 카테고리: missing_values
# 기법 이름: custom_imputation
# 코드: # 사용자 정의 결측값 처리 코드
# 설명: 사용자 정의 결측값 처리 방법
# 사용 사례: 특별한 도메인 지식이 필요한 경우
# 키워드: custom, domain, knowledge
```

### Knowledge Base 확장

#### 1. 새로운 카테고리 추가
```python
# manage_kb.py에서 기법 추가 시 새로운 카테고리 이름 입력
# 자동으로 새 카테고리가 생성됩니다
```

#### 2. 기존 카테고리 확장
```python
# 기존 카테고리에 새로운 기법 추가
# 예: missing_values 카테고리에 새로운 결측값 처리 방법 추가
```

#### 3. 임베딩 업데이트
```bash
# Knowledge Base 변경 후 반드시 임베딩 재생성
python KB_rag_agents/manage_kb.py
# 메뉴에서 7번 선택 (임베딩 재생성)
```

## 사용 예시

### RAG 기반 전처리 워크플로우

```python
from planning_agents.planning_agent import planning_agent
from KB_rag_agents.KB_rag_agent import kb_rag_agent

# 1. EDA 결과 준비
eda_results = {
    'text_analysis': '데이터 분석 결과...',
    'null_analysis_text': '결측값 분석...',
    'outlier_analysis_text': '이상치 분석...',
    # ... 기타 분석 결과
}

# 2. Planning Agent 실행
planning_result = planning_agent.invoke(eda_results)

# 3. KB RAG Agent 실행
rag_result = kb_rag_agent.invoke({
    **eda_results,
    'planning_info': planning_result['planning_info']
})

# 4. 생성된 코드 사용
generated_code = rag_result['generated_preprocessing_code']
```

## 주요 특징

### 1. 지식 기반 접근
- 검증된 전처리 코드 예제 데이터베이스
- 유사도 기반 관련 코드 검색
- 일관성 있는 코드 생성

### 2. 계획 기반 처리
- EDA 결과 분석을 통한 전처리 계획 수립
- 작업 우선순위 및 의존성 고려
- 체계적인 전처리 파이프라인

### 3. 멀티모달 LLM 활용
- 텍스트와 이미지 모두 활용
- OpenAI GPT-4o-mini 모델 사용
- 임베딩 기반 유사도 검색

### 4. 확장 가능한 구조
- 새로운 전처리 기법 쉽게 추가
- Knowledge Base 지속적 확장
- 모듈화된 에이전트 구조

### 5. Knowledge Base 관리
- JSON 기반 구조화된 Knowledge Base
- 관리 도구를 통한 쉬운 확장
- 임베딩 기반 유사도 검색

## 기술 스택

- **LLM**: OpenAI GPT-4o-mini
- **Embedding**: OpenAI text-embedding-3-small
- **RAG**: Cosine Similarity 기반 검색
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Framework**: LangChain, LangGraph

## 라이선스

MIT License 