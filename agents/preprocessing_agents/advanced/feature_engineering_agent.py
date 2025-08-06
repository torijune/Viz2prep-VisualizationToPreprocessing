"""
특성 엔지니어링 전처리 에이전트
새로운 특성을 생성하고 기존 특성을 변환하는 다양한 방법을 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def engineer_features(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    특성 엔지니어링을 수행하는 전처리 함수
    
    Args:
        inputs: DataFrame과 엔지니어링 방법이 포함된 입력 딕셔너리
        
    Returns:
        새로운 특성이 추가된 DataFrame이 포함된 딕셔너리
    """
    df = inputs["dataframe"].copy()
    methods = inputs.get("engineering_methods", ["auto"])  # auto, polynomial, interaction, datetime, binning
    target_column = inputs.get("target_column", None)
    
    # EDA 결과물들 가져오기
    numeric_analysis_text = inputs.get("numeric_analysis_text", "")
    numeric_image_paths = inputs.get("numeric_image_paths", [])
    cate_analysis_text = inputs.get("cate_analysis_text", "")
    corr_analysis_text = inputs.get("corr_analysis_text", "")
    text_analysis = inputs.get("text_analysis", "")
    
    if isinstance(methods, str):
        methods = [methods]
    
    print(f"특성 엔지니어링 시작: {methods} 방법")
    print(f"  원본 특성 수: {len(df.columns)}")
    
    # MultiModal LLM을 사용한 전처리 코드 생성
    if "auto" in methods:
        preprocessing_code = generate_feature_engineering_code_with_llm(
            df, target_column, numeric_analysis_text, numeric_image_paths,
            cate_analysis_text, corr_analysis_text, text_analysis
        )
        
        # 생성된 코드 실행
        try:
            exec(preprocessing_code)
            print("LLM 생성 코드로 특성 엔지니어링 완료")
        except Exception as e:
            print(f"LLM 생성 코드 실행 오류: {e}")
            # 폴백: 기본 자동 처리
            df = apply_basic_feature_engineering(df, target_column)
    else:
        # 수동 방법 사용
        df = apply_manual_feature_engineering(df, methods, target_column, inputs)
    
    # 엔지니어링 정보 생성
    engineering_info = {
        'original_features': df.columns.tolist(),
        'methods': methods,
        'new_features': [],
        'final_features': df.columns.tolist(),
        'total_new_features': len(df.columns) - len(df.columns)
    }
    
    print(f"특성 엔지니어링 완료: 최종 {len(df.columns)}개 특성")
    
    return {
        **inputs,
        "dataframe": df,
        "engineering_info": engineering_info
    }


def generate_feature_engineering_code_with_llm(df: pd.DataFrame, target_column: Optional[str],
                                            numeric_analysis_text: str, numeric_image_paths: List[str],
                                            cate_analysis_text: str, corr_analysis_text: str,
                                            text_analysis: str) -> str:
    """
    MultiModal LLM을 사용하여 특성 엔지니어링 코드를 생성합니다.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=2000
    )
    
    # 데이터 특성 분석
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    # 특성 정보 요약
    feature_summary = []
    feature_summary.append(f"수치형 변수: {len(numeric_cols)}개 - {numeric_cols}")
    feature_summary.append(f"범주형 변수: {len(categorical_cols)}개 - {categorical_cols}")
    feature_summary.append(f"날짜 변수: {len(datetime_cols)}개 - {datetime_cols}")
    
    prompt = f"""
You are a data preprocessing expert. Please write Python code to perform feature engineering based on the following information.

=== Dataset Information ===
- Data size: {df.shape[0]} rows x {df.shape[1]} columns
- Target variable: {target_column or 'None'}
- Dataset head:
{df.head().to_string()}

=== Feature Information ===
{chr(10).join(feature_summary)}

=== Numeric Variable Analysis ===
{numeric_analysis_text}

=== Categorical Variable Analysis ===
{cate_analysis_text}

=== Correlation Analysis ===
{corr_analysis_text}

=== Overall Data Analysis ===
{text_analysis}

=== Requirements ===
1. Create polynomial features if there are 2 or more numeric variables
2. Create one-hot encoding or frequency features for categorical variables
3. Create time features if date variables exist
4. Code must be executable

Please write code in the following format:
```python
# Feature engineering code
# df is an already defined DataFrame
```

Return only the code without explanations.
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        code = response.content
        
        # 코드 블록에서 실제 코드만 추출
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        
        return code
    except Exception as e:
        print(f"LLM 코드 생성 오류: {e}")
        return ""


def apply_basic_feature_engineering(df: pd.DataFrame, target_column: Optional[str]) -> pd.DataFrame:
    """
    기본 특성 엔지니어링 방법을 적용합니다.
    """
    # 수치형 변수 처리
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    if len(numeric_cols) >= 2:
        # 2차 다항식 특성 생성
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                feature_name = f"{col1}_{col2}_interaction"
                df[feature_name] = df[col1] * df[col2]
        
        # 제곱 특성 생성
        for col in numeric_cols:
            feature_name = f"{col}_squared"
            df[feature_name] = df[col] ** 2
        
        print(f"    → 다항식 특성 생성: {len(numeric_cols)}개 변수에서 특성 생성")
    
    # 범주형 변수 처리
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        if df[col].nunique() <= 10:  # 고유값이 10개 이하인 경우만
            # 빈도 기반 특성
            value_counts = df[col].value_counts()
            feature_name = f"{col}_frequency"
            df[feature_name] = df[col].map(value_counts)
    
    # 날짜 변수 처리
    datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    for col in datetime_cols:
        try:
            df[col] = pd.to_datetime(df[col])
            
            # 시간 관련 특성 생성
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df[f"{col}_quarter"] = df[col].dt.quarter
            
            print(f"    → 날짜 특성 생성: {col}에서 5개 특성 생성")
        except Exception as e:
            print(f"    ⚠️  {col} 날짜 변환 실패: {e}")
    
    return df


def apply_manual_feature_engineering(df: pd.DataFrame, methods: List[str], 
                                   target_column: Optional[str], inputs: Dict) -> pd.DataFrame:
    """
    수동 특성 엔지니어링 방법을 적용합니다.
    """
    for method in methods:
        if method == "polynomial":
            df = apply_polynomial_features(df, target_column)
        elif method == "interaction":
            df = apply_interaction_features(df, target_column)
        elif method == "datetime":
            df = apply_datetime_features(df)
        elif method == "binning":
            df = apply_binning_features(df, target_column)
        elif method == "categorical":
            df = apply_categorical_features(df)
        elif method == "statistical":
            df = apply_statistical_features(df, target_column)
    
    return df


def apply_polynomial_features(df: pd.DataFrame, target_column: Optional[str]) -> pd.DataFrame:
    """
    다항식 특성 생성
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    if len(numeric_cols) >= 2:
        # 2차 다항식 특성 생성
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                feature_name = f"{col1}_{col2}_interaction"
                df[feature_name] = df[col1] * df[col2]
        
        # 제곱 특성 생성
        for col in numeric_cols:
            feature_name = f"{col}_squared"
            df[feature_name] = df[col] ** 2
        
        print(f"    → 다항식 특성 생성: {len(numeric_cols)}개 변수에서 특성 생성")
    
    return df


def apply_interaction_features(df: pd.DataFrame, target_column: Optional[str]) -> pd.DataFrame:
    """
    상호작용 특성 생성
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    if len(numeric_cols) >= 2:
        # 모든 수치형 변수 쌍의 상호작용
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                feature_name = f"{col1}_{col2}_interaction"
                df[feature_name] = df[col1] * df[col2]
        
        print(f"    → 상호작용 특성 생성: {len(numeric_cols)}개 변수에서 특성 생성")
    
    return df


def apply_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    날짜/시간 특성 생성
    """
    datetime_cols = []
    
    # 날짜 컬럼 찾기
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].iloc[0])
                datetime_cols.append(col)
            except:
                continue
    
    for col in datetime_cols:
        try:
            df[col] = pd.to_datetime(df[col])
            
            # 시간 관련 특성 생성
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df[f"{col}_quarter"] = df[col].dt.quarter
            
            print(f"    → 날짜 특성 생성: {col}에서 5개 특성 생성")
        except Exception as e:
            print(f"    ⚠️  {col} 날짜 변환 실패: {e}")
    
    return df


def apply_binning_features(df: pd.DataFrame, target_column: Optional[str]) -> pd.DataFrame:
    """
    구간화 특성 생성
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    for col in numeric_cols:
        # 5개 구간으로 나누기
        feature_name = f"{col}_binned"
        df[feature_name] = pd.cut(df[col], bins=5, labels=False)
    
    print(f"    → 구간화 특성 생성: {len(numeric_cols)}개 변수에서 특성 생성")
    return df


def apply_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    범주형 변수 처리
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in categorical_cols:
        if df[col].nunique() <= 10:  # 고유값이 10개 이하인 경우만
            # 빈도 기반 특성
            value_counts = df[col].value_counts()
            feature_name = f"{col}_frequency"
            df[feature_name] = df[col].map(value_counts)
            
            # 원핫 인코딩 (상위 5개 값만)
            top_values = value_counts.head(5).index
            for value in top_values:
                feature_name = f"{col}_{value}_onehot"
                df[feature_name] = (df[col] == value).astype(int)
    
    print(f"    → 범주형 특성 생성: {len(categorical_cols)}개 변수에서 특성 생성")
    return df


def apply_statistical_features(df: pd.DataFrame, target_column: Optional[str]) -> pd.DataFrame:
    """
    통계적 특성 생성
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    for col in numeric_cols:
        # Z-score
        feature_name = f"{col}_zscore"
        df[feature_name] = (df[col] - df[col].mean()) / df[col].std()
        
        # 로그 변환 (양수 값만)
        if (df[col] > 0).all():
            feature_name = f"{col}_log"
            df[feature_name] = np.log(df[col])
    
    print(f"    → 통계적 특성 생성: {len(numeric_cols)}개 변수에서 특성 생성")
    return df


# LangGraph 노드로 사용할 수 있는 함수
feature_engineering_agent = RunnableLambda(engineer_features)