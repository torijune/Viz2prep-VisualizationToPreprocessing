#!/usr/bin/env python3
"""
전문 코더 노드들
각 도메인별로 전문화된 전처리 코드를 생성하는 노드들
"""

import os
import sys
from typing import Dict, Any
import json
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Knowledge Base 에이전트 import
from KB_rag_agents.KB_rag_agent import KnowledgeBaseRAGAgent

from workflow_state import WorkflowState

def numeric_coder_node(state: WorkflowState) -> WorkflowState:
    """
    수치형 전처리 코더 노드
    
    🔍 INPUT STATES:
    - numeric_plan: 수치형 전처리 계획
    - numeric_eda_result: 수치형 변수 EDA 결과
    
    📊 OUTPUT STATES:
    - numeric_code: 수치형 전처리 코드
    
    ➡️ NEXT EDGE: executor (1번째 순서)
    """
    print("💻 [CODE] 수치형 전처리 코드 생성 중...")
    
    try:
        numeric_plan = state.get("numeric_plan", {})
        numeric_eda = state.get("numeric_eda_result", {})
        dataframe = state.get("dataframe")
        
        # 실제 데이터프레임의 수치형 컬럼 가져오기
        actual_numeric_columns = dataframe.select_dtypes(include=['number']).columns.tolist()
        
        # Knowledge Base에서 관련 코드 검색
        kb_agent = KnowledgeBaseRAGAgent()
        kb_query = f"수치형 변수 전처리 {' '.join(numeric_plan.get('techniques', []))}"
        kb_result = kb_agent.search_techniques(kb_query, numeric_plan.get('techniques', []))
        
        # LLM을 사용하여 코드 생성
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=1500)
        
        prompt = f"""
당신은 수치형 데이터 전처리 코드 생성 전문가입니다.
아래 계획과 EDA 결과를 바탕으로 실행 가능한 Python 코드를 생성해주세요.

=== 실제 데이터프레임 정보 ===
- 전체 컬럼: {list(dataframe.columns)}
- 수치형 컬럼: {actual_numeric_columns}
- 데이터프레임 크기: {dataframe.shape}

=== 수치형 전처리 계획 ===
기법: {numeric_plan.get('techniques', [])}
근거: {numeric_plan.get('rationale', '')}

=== EDA 결과 ===
분석된 컬럼: {numeric_eda.get('columns_analyzed', [])}
통계: {numeric_eda.get('statistics', {})}

=== Knowledge Base 참고 ===
{kb_result}

=== 요구사항 ===
1. 함수명: def preprocess_numeric_data(df):
2. 입력: pandas DataFrame (사용자가 제공한 실제 데이터프레임)
3. 출력: 전처리된 pandas DataFrame
4. 반드시 실행 가능한 코드여야 함
5. 필요한 import 구문 포함
6. 오류 처리 포함
7. 실제 데이터프레임의 컬럼명을 사용하세요: {actual_numeric_columns}
8. ⚠️ 중요: 새로운 데이터프레임을 정의하지 마세요 (df = pd.DataFrame(...) 금지)
9. ⚠️ 중요: 입력받은 df를 직접 사용하여 전처리하세요
10. ⚠️ 중요: Iris 데이터의 실제 컬럼명 사용: {list(dataframe.columns)}
11. ⚠️ 중요: 필요한 모든 import 구문을 포함하세요 (pandas, numpy, sklearn, scipy 등)

Python 코드만 생성해주세요 (주석 포함):
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        generated_code = response.content
        
        # 코드가 ```python으로 둘러싸여 있다면 제거
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0].strip()
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].strip()
        
        print(f"✅ [CODE] 수치형 코드 생성 완료 - {len(numeric_plan.get('techniques', []))}개 기법")
        print("📝 [CODE] 생성된 수치형 전처리 코드:")
        print("=" * 50)
        print(generated_code)
        print("=" * 50)
        
        return {
            **state,
            "numeric_code": generated_code
        }
        
    except Exception as e:
        print(f"❌ [CODE] 수치형 코드 생성 오류: {e}")
        # 기본 코드 제공
        default_code = """
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import mstats

def preprocess_numeric_data(df):
    \"\"\"수치형 데이터 전처리 (기본)\"\"\"
    try:
        df_processed = df.copy()
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            scaler = StandardScaler()
            df_processed[numeric_columns] = scaler.fit_transform(df_processed[numeric_columns])
        
        return df_processed
    except Exception as e:
        print(f"수치형 전처리 오류: {e}")
        return df
"""
        return {
            **state,
            "numeric_code": default_code
        }


def category_coder_node(state: WorkflowState) -> WorkflowState:
    """
    범주형 전처리 코더 노드
    
    🔍 INPUT STATES:
    - category_plan: 범주형 전처리 계획
    - category_eda_result: 범주형 변수 EDA 결과
    
    📊 OUTPUT STATES:
    - category_code: 범주형 전처리 코드
    
    ➡️ NEXT EDGE: executor (2번째 순서)
    """
    print("💻 [CODE] 범주형 전처리 코드 생성 중...")
    
    try:
        category_plan = state.get("category_plan", {})
        category_eda = state.get("category_eda_result", {})
        dataframe = state.get("dataframe")
        
        # 실제 데이터프레임의 범주형 컬럼 가져오기
        actual_categorical_columns = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Knowledge Base에서 관련 코드 검색
        kb_agent = KnowledgeBaseRAGAgent()
        kb_query = f"범주형 변수 전처리 {' '.join(category_plan.get('techniques', []))}"
        kb_result = kb_agent.search_techniques(kb_query, category_plan.get('techniques', []))
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=1500)
        
        prompt = f"""
당신은 범주형 데이터 전처리 코드 생성 전문가입니다.
아래 계획과 EDA 결과를 바탕으로 실행 가능한 Python 코드를 생성해주세요.

=== 실제 데이터프레임 정보 ===
- 전체 컬럼: {list(dataframe.columns)}
- 범주형 컬럼: {actual_categorical_columns}
- 데이터프레임 크기: {dataframe.shape}

=== 범주형 전처리 계획 ===
기법: {category_plan.get('techniques', [])}
근거: {category_plan.get('rationale', '')}

=== EDA 결과 ===
분석된 컬럼: {category_eda.get('columns_analyzed', [])}
범주형 요약: {category_eda.get('categorical_summary', {})}

=== Knowledge Base 참고 ===
{kb_result}

=== 요구사항 ===
1. 함수명: def preprocess_categorical_data(df):
2. 입력: pandas DataFrame (사용자가 제공한 실제 데이터프레임)
3. 출력: 전처리된 pandas DataFrame
4. 반드시 실행 가능한 코드여야 함
5. 필요한 import 구문 포함
6. 오류 처리 포함
7. 실제 데이터프레임의 컬럼명을 사용하세요: {actual_categorical_columns}
8. ⚠️ 중요: 새로운 데이터프레임을 정의하지 마세요 (df = pd.DataFrame(...) 금지)
9. ⚠️ 중요: 입력받은 df를 직접 사용하여 전처리하세요
10. ⚠️ 중요: Iris 데이터의 실제 컬럼명 사용: {list(dataframe.columns)}
11. ⚠️ 중요: 범주형 변수는 원핫 인코딩(pd.get_dummies)을 사용하세요. Label Encoding 대신 원핫 인코딩을 사용하세요.
12. ⚠️ 중요: 원핫 인코딩 시 dtype=int를 사용하여 0/1 값으로 변환하세요.
13. ⚠️ 중요: drop_first=False를 사용하여 모든 카테고리를 유지하세요.

Python 코드만 생성해주세요 (주석 포함):
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        generated_code = response.content
        
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0].strip()
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].strip()
        
        print(f"✅ [CODE] 범주형 코드 생성 완료 - {len(category_plan.get('techniques', []))}개 기법")
        print("📝 [CODE] 생성된 범주형 전처리 코드:")
        print("=" * 50)
        print(generated_code)
        print("=" * 50)
        
        return {
            **state,
            "category_code": generated_code
        }
        
    except Exception as e:
        print(f"❌ [CODE] 범주형 코드 생성 오류: {e}")
        default_code = """
import pandas as pd

def preprocess_categorical_data(df):
    \"\"\"범주형 데이터 전처리 (기본 - 원핫 인코딩)\"\"\"
    try:
        df_processed = df.copy()
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if df_processed[col].nunique() <= 10:
                # 원핫 인코딩 사용 (int 타입으로 변환, 모든 카테고리 유지)
                df_processed = pd.get_dummies(df_processed, columns=[col], drop_first=False, dtype=int)
            else:
                # 원핫 인코딩 (상위 5개 카테고리만)
                top_categories = df_processed[col].value_counts().head(5).index
                for cat in top_categories:
                    df_processed[f'{col}_{cat}'] = (df_processed[col] == cat).astype(int)
                df_processed.drop(col, axis=1, inplace=True)
        
        return df_processed
    except Exception as e:
        print(f"범주형 전처리 오류: {e}")
        return df
"""
        return {
            **state,
            "category_code": default_code
        }


def outlier_coder_node(state: WorkflowState) -> WorkflowState:
    """
    이상치 전처리 코더 노드
    
    🔍 INPUT STATES:
    - outlier_plan: 이상치 전처리 계획
    - outlier_eda_result: 이상치 EDA 결과
    
    📊 OUTPUT STATES:
    - outlier_code: 이상치 전처리 코드
    
    ➡️ NEXT EDGE: executor (3번째 순서)
    """
    print("💻 [CODE] 이상치 전처리 코드 생성 중...")
    
    try:
        outlier_plan = state.get("outlier_plan", {})
        outlier_eda = state.get("outlier_eda_result", {})
        dataframe = state.get("dataframe")
        
        # 실제 데이터프레임의 수치형 컬럼 가져오기 (이상치는 수치형에만 적용)
        actual_numeric_columns = dataframe.select_dtypes(include=['number']).columns.tolist()
        
        kb_agent = KnowledgeBaseRAGAgent()
        kb_query = f"이상치 처리 {' '.join(outlier_plan.get('techniques', []))}"
        kb_result = kb_agent.search_techniques(kb_query, outlier_plan.get('techniques', []))
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=1500)
        
        prompt = f"""
당신은 이상치 처리 코드 생성 전문가입니다.
아래 계획과 EDA 결과를 바탕으로 실행 가능한 Python 코드를 생성해주세요.

=== 실제 데이터프레임 정보 ===
- 전체 컬럼: {list(dataframe.columns)}
- 수치형 컬럼 (이상치 처리 대상): {actual_numeric_columns}
- 데이터프레임 크기: {dataframe.shape}

=== 이상치 전처리 계획 ===
기법: {outlier_plan.get('techniques', [])}
근거: {outlier_plan.get('rationale', '')}

=== EDA 결과 ===
분석된 컬럼: {outlier_eda.get('columns_analyzed', [])}
총 이상치: {outlier_eda.get('total_outliers', 0)}개

=== Knowledge Base 참고 ===
{kb_result}

=== 요구사항 ===
1. 함수명: def preprocess_outliers(df):
2. 입력: pandas DataFrame (사용자가 제공한 실제 데이터프레임)
3. 출력: 전처리된 pandas DataFrame
4. 반드시 실행 가능한 코드여야 함
5. 필요한 import 구문 포함
6. 오류 처리 포함
7. 실제 데이터프레임의 수치형 컬럼명을 사용하세요: {actual_numeric_columns}
8. ⚠️ 중요: 새로운 데이터프레임을 정의하지 마세요 (df = pd.DataFrame(...) 금지)
9. ⚠️ 중요: 입력받은 df를 직접 사용하여 전처리하세요
10. ⚠️ 중요: Iris 데이터의 실제 컬럼명 사용: {list(dataframe.columns)}

Python 코드만 생성해주세요 (주석 포함):
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        generated_code = response.content
        
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0].strip()
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].strip()
        
        print(f"✅ [CODE] 이상치 코드 생성 완료 - {len(outlier_plan.get('techniques', []))}개 기법")
        print("📝 [CODE] 생성된 이상치 전처리 코드:")
        print("=" * 50)
        print(generated_code)
        print("=" * 50)
        
        return {
            **state,
            "outlier_code": generated_code
        }
        
    except Exception as e:
        print(f"❌ [CODE] 이상치 코드 생성 오류: {e}")
        default_code = """
import pandas as pd
import numpy as np

def preprocess_outliers(df):
    \"\"\"이상치 처리 (기본 - IQR 방법)\"\"\"
    try:
        df_processed = df.copy()
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 이상치 제한 (capping)
            df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_processed
    except Exception as e:
        print(f"이상치 전처리 오류: {e}")
        return df
"""
        return {
            **state,
            "outlier_code": default_code
        }


def nulldata_coder_node(state: WorkflowState) -> WorkflowState:
    """
    결측값 전처리 코더 노드
    
    🔍 INPUT STATES:
    - nulldata_plan: 결측값 전처리 계획
    - nulldata_eda_result: 결측값 EDA 결과
    
    📊 OUTPUT STATES:
    - nulldata_code: 결측값 전처리 코드
    
    ➡️ NEXT EDGE: executor (4번째 순서)
    """
    print("💻 [CODE] 결측값 전처리 코드 생성 중...")
    
    try:
        nulldata_plan = state.get("nulldata_plan", {})
        nulldata_eda = state.get("nulldata_eda_result", {})
        dataframe = state.get("dataframe")
        
        # 실제 데이터프레임의 결측값이 있는 컬럼 가져오기
        actual_missing_columns = dataframe.columns[dataframe.isnull().any()].tolist()
        
        kb_agent = KnowledgeBaseRAGAgent()
        kb_query = f"결측값 처리 {' '.join(nulldata_plan.get('techniques', []))}"
        kb_result = kb_agent.search_techniques(kb_query, nulldata_plan.get('techniques', []))
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=1500)
        
        prompt = f"""
당신은 결측값 처리 코드 생성 전문가입니다.
아래 계획과 EDA 결과를 바탕으로 실행 가능한 Python 코드를 생성해주세요.

=== 실제 데이터프레임 정보 ===
- 전체 컬럼: {list(dataframe.columns)}
- 결측값이 있는 컬럼: {actual_missing_columns}
- 데이터프레임 크기: {dataframe.shape}
- 총 결측값 수: {dataframe.isnull().sum().sum()}개

=== 결측값 전처리 계획 ===
기법: {nulldata_plan.get('techniques', [])}
근거: {nulldata_plan.get('rationale', '')}

=== EDA 결과 ===
결측값 컬럼: {nulldata_eda.get('columns_with_nulls', [])}
총 결측값: {nulldata_eda.get('total_missing', 0)}개

=== Knowledge Base 참고 ===
{kb_result}

=== 요구사항 ===
1. 함수명: def preprocess_missing_data(df):
2. 입력: pandas DataFrame (사용자가 제공한 실제 데이터프레임)
3. 출력: 전처리된 pandas DataFrame
4. 반드시 실행 가능한 코드여야 함
5. 필요한 import 구문 포함
6. 오류 처리 포함
7. 실제 데이터프레임의 컬럼명을 사용하세요: {list(dataframe.columns)}
8. ⚠️ 중요: 새로운 데이터프레임을 정의하지 마세요 (df = pd.DataFrame(...) 금지)
9. ⚠️ 중요: 입력받은 df를 직접 사용하여 전처리하세요
10. ⚠️ 중요: Iris 데이터의 실제 컬럼명 사용: {list(dataframe.columns)}

Python 코드만 생성해주세요 (주석 포함):
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        generated_code = response.content
        
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0].strip()
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].strip()
        
        print(f"✅ [CODE] 결측값 코드 생성 완료 - {len(nulldata_plan.get('techniques', []))}개 기법")
        print("📝 [CODE] 생성된 결측값 전처리 코드:")
        print("=" * 50)
        print(generated_code)
        print("=" * 50)
        
        return {
            **state,
            "nulldata_code": generated_code
        }
        
    except Exception as e:
        print(f"❌ [CODE] 결측값 코드 생성 오류: {e}")
        default_code = """
import pandas as pd
import numpy as np

def preprocess_missing_data(df):
    \"\"\"결측값 처리 (기본)\"\"\"
    try:
        df_processed = df.copy()
        
        # 결측값이 있는 컬럼들 확인
        missing_columns = df_processed.columns[df_processed.isnull().any()].tolist()
        
        if len(missing_columns) == 0:
            print("결측값이 없습니다.")
            return df_processed
        
        print(f"결측값 처리 중: {len(missing_columns)}개 컬럼")
        
        for col in missing_columns:
            missing_count = df_processed[col].isnull().sum()
            print(f"  - {col}: {missing_count}개 결측값 처리")
            
            if df_processed[col].dtype in ['object', 'category']:
                # 범주형: 최빈값으로 채우기
                mode_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown'
                df_processed[col].fillna(mode_value, inplace=True)
                print(f"    → 최빈값 '{mode_value}'로 채움")
            else:
                # 수치형: 중앙값으로 채우기
                median_value = df_processed[col].median()
                df_processed[col].fillna(median_value, inplace=True)
                print(f"    → 중앙값 {median_value}로 채움")
        
        print(f"결측값 처리 완료: {df_processed.isnull().sum().sum()}개 남음")
        return df_processed
        
    except Exception as e:
        print(f"결측값 전처리 오류: {e}")
        return df
"""
        return {
            **state,
            "nulldata_code": default_code
        }


def corr_coder_node(state: WorkflowState) -> WorkflowState:
    """
    상관관계 전처리 코더 노드
    
    🔍 INPUT STATES:
    - corr_plan: 상관관계 전처리 계획
    - corr_eda_result: 상관관계 EDA 결과
    
    📊 OUTPUT STATES:
    - corr_code: 상관관계 전처리 코드
    
    ➡️ NEXT EDGE: executor (5번째 순서)
    """
    print("💻 [CODE] 상관관계 전처리 코드 생성 중...")
    
    try:
        corr_plan = state.get("corr_plan", {})
        corr_eda = state.get("corr_eda_result", {})
        dataframe = state.get("dataframe")
        
        # 실제 데이터프레임의 수치형 컬럼 가져오기 (상관관계는 수치형에만 적용)
        actual_numeric_columns = dataframe.select_dtypes(include=['number']).columns.tolist()
        
        kb_agent = KnowledgeBaseRAGAgent()
        kb_query = f"상관관계 특성선택 {' '.join(corr_plan.get('techniques', []))}"
        kb_result = kb_agent.search_techniques(kb_query, corr_plan.get('techniques', []))
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=1500)
        
        prompt = f"""
당신은 상관관계 기반 특성 선택 코드 생성 전문가입니다.
아래 계획과 EDA 결과를 바탕으로 실행 가능한 Python 코드를 생성해주세요.

=== 실제 데이터프레임 정보 ===
- 전체 컬럼: {list(dataframe.columns)}
- 수치형 컬럼 (상관관계 분석 대상): {actual_numeric_columns}
- 데이터프레임 크기: {dataframe.shape}

=== 상관관계 전처리 계획 ===
기법: {corr_plan.get('techniques', [])}
근거: {corr_plan.get('rationale', '')}

=== EDA 결과 ===
강한 상관관계: {corr_eda.get('high_correlations', [])}

=== Knowledge Base 참고 ===
{kb_result}

=== 요구사항 ===
1. 함수명: def preprocess_correlation_features(df):
2. 입력: pandas DataFrame (사용자가 제공한 실제 데이터프레임)
3. 출력: 전처리된 pandas DataFrame
4. 반드시 실행 가능한 코드여야 함
5. 필요한 import 구문 포함
6. 오류 처리 포함
7. 실제 데이터프레임의 수치형 컬럼명을 사용하세요: {actual_numeric_columns}
8. ⚠️ 중요: 새로운 데이터프레임을 정의하지 마세요 (df = pd.DataFrame(...) 금지)
9. ⚠️ 중요: 입력받은 df를 직접 사용하여 전처리하세요
10. ⚠️ 중요: Iris 데이터의 실제 컬럼명 사용: {list(dataframe.columns)}

Python 코드만 생성해주세요 (주석 포함):
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        generated_code = response.content
        
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0].strip()
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].strip()
        
        print(f"✅ [CODE] 상관관계 코드 생성 완료 - {len(corr_plan.get('techniques', []))}개 기법")
        print("📝 [CODE] 생성된 상관관계 전처리 코드:")
        print("=" * 50)
        print(generated_code)
        print("=" * 50)
        
        return {
            **state,
            "corr_code": generated_code
        }
        
    except Exception as e:
        print(f"❌ [CODE] 상관관계 코드 생성 오류: {e}")
        default_code = """
import pandas as pd
import numpy as np

def preprocess_correlation_features(df):
    \"\"\"상관관계 기반 특성 선택 (기본)\"\"\"
    try:
        df_processed = df.copy()
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 1:
            # 높은 상관관계 특성 제거 (0.95 이상)
            corr_matrix = df_processed[numeric_columns].corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
            df_processed.drop(columns=to_drop, inplace=True)
        
        return df_processed
    except Exception as e:
        print(f"상관관계 전처리 오류: {e}")
        return df
"""
        return {
            **state,
            "corr_code": default_code
        } 

def code_debug_agent(state: WorkflowState) -> WorkflowState:
    """
    코드 디버깅 에이전트
    전처리 코드 실행 중 발생한 오류를 LLM을 사용하여 수정합니다.
    
    🔍 INPUT STATES:
    - error_message: 발생한 오류 메시지
    - original_code: 원본 전처리 코드
    - dataframe_info: 데이터프레임 정보
    - preprocessing_plan: 전처리 계획
    
    📊 OUTPUT STATES:
    - fixed_code: 수정된 전처리 코드
    - debug_info: 디버깅 정보
    
    ➡️ NEXT EDGE: 다시 전처리 실행
    """
    print("🔧 [DEBUG] 코드 디버깅 시작...")
    
    try:
        error_message = state.get("error_message", "")
        original_code = state.get("original_code", "")
        dataframe_info = state.get("dataframe_info", {})
        preprocessing_plan = state.get("preprocessing_plan", {})
        
        # LLM을 사용하여 코드 수정
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=2000)
        
        prompt = f"""
당신은 Python 코드 디버깅 전문가입니다.
전처리 코드 실행 중 발생한 오류를 분석하고 수정해주세요.

=== 오류 메시지 ===
{error_message}

=== 원본 전처리 코드 ===
{original_code}

=== 데이터프레임 정보 ===
{json.dumps(dataframe_info, indent=2, ensure_ascii=False)}

=== 전처리 계획 ===
{json.dumps(preprocessing_plan, indent=2, ensure_ascii=False)}

=== 디버깅 지침 ===
1. **오류 분석**: 오류의 원인을 정확히 파악하세요
2. **코드 검토**: 원본 코드의 문제점을 찾으세요
3. **수정 방안**: 구체적인 수정 방법을 제시하세요
4. **코드 수정**: 수정된 완전한 코드를 제공하세요

=== 수정 시 고려사항 ===
- 컬럼명이 실제 데이터프레임에 존재하는지 확인
- 데이터 타입이 올바른지 확인
- 필요한 라이브러리가 import되어 있는지 확인
- 변수명과 함수명이 올바른지 확인
- 예외 처리가 적절한지 확인

=== 응답 형식 ===
{{
    "error_analysis": "오류 원인 분석",
    "fix_strategy": "수정 전략",
    "fixed_code": "수정된 완전한 코드",
    "debug_info": {{
        "original_error": "원본 오류",
        "fix_description": "수정 내용 설명",
        "prevention_tips": "향후 예방 방법"
    }}
}}

JSON 형식으로만 응답해주세요.
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        
        # JSON 파싱
        try:
            content = response.content.strip()
            if content.startswith('```json'):
                content = content.split('```json')[1].split('```')[0].strip()
            elif content.startswith('```'):
                content = content.split('```')[1].split('```')[0].strip()
            
            result = json.loads(content)
        except (json.JSONDecodeError, IndexError) as e:
            print(f"⚠️ [DEBUG] JSON 파싱 오류: {e}")
            result = {
                "error_analysis": "JSON 파싱 실패",
                "fix_strategy": "기본 수정 전략",
                "fixed_code": original_code,
                "debug_info": {
                    "original_error": error_message,
                    "fix_description": "파싱 오류로 인한 기본 수정",
                    "prevention_tips": "JSON 응답 형식 확인 필요"
                }
            }
        
        debug_info = {
            "error_analysis": result.get("error_analysis", ""),
            "fix_strategy": result.get("fix_strategy", ""),
            "debug_info": result.get("debug_info", {}),
            "original_error": error_message,
            "original_code": original_code
        }
        
        print(f"✅ [DEBUG] 코드 디버깅 완료")
        
        # 디버깅 결과 로깅
        print("\n" + "="*50)
        print("🔧 [DEBUG] 코드 디버깅 결과")
        print("="*50)
        print(f"📊 오류 분석: {debug_info['error_analysis']}")
        print(f"🔧 수정 전략: {debug_info['fix_strategy']}")
        print(f"💡 예방 방법: {debug_info['debug_info'].get('prevention_tips', 'N/A')}")
        
        return {
            **state,
            "fixed_code": result.get("fixed_code", original_code),
            "debug_info": debug_info,
            "debug_status": "success"
        }
        
    except Exception as e:
        print(f"❌ [DEBUG] 코드 디버깅 오류: {e}")
        return {
            **state,
            "fixed_code": original_code,
            "debug_info": {
                "error_analysis": f"디버깅 자체 오류: {str(e)}",
                "fix_strategy": "기본 코드 유지",
                "debug_info": {
                    "original_error": error_message,
                    "fix_description": "디버깅 실패로 원본 코드 유지",
                    "prevention_tips": "디버깅 시스템 점검 필요"
                }
            },
            "debug_status": "error"
        } 