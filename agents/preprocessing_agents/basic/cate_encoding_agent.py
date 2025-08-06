"""
범주형 변수 인코딩 전처리 에이전트
범주형 변수를 수치형으로 변환하는 다양한 인코딩 방법을 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def encode_categorical(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    범주형 변수를 인코딩하는 전처리 함수
    
    Args:
        inputs: DataFrame과 인코딩 방법이 포함된 입력 딕셔너리
        
    Returns:
        인코딩된 DataFrame이 포함된 딕셔너리
    """
    df = inputs["dataframe"].copy()
    method = inputs.get("encoding_method", "auto")  # auto, label, onehot, ordinal, target
    columns = inputs.get("encoding_columns", None)  # 특정 컬럼만 인코딩
    
    # EDA 결과물들 가져오기
    cate_analysis_text = inputs.get("cate_analysis_text", "")
    cate_image_paths = inputs.get("cate_image_paths", [])
    text_analysis = inputs.get("text_analysis", "")
    
    # 범주형 컬럼 선택
    if columns is None:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        categorical_columns = [col for col in columns if col in df.columns]
    
    if not categorical_columns:
        print("범주형 인코딩: 범주형 변수가 없습니다.")
        return {
            **inputs,
            "dataframe": df,
            "encoding_info": {}
        }
    
    print(f"범주형 인코딩 시작: {method} 방법")
    print(f"  대상 컬럼: {categorical_columns}")
    
    # MultiModal LLM을 사용한 전처리 코드 생성
    if method == "auto":
        preprocessing_code = generate_categorical_encoding_code_with_llm(
            df, categorical_columns, cate_analysis_text, cate_image_paths, text_analysis
        )
        
        # 생성된 코드 실행
        try:
            exec(preprocessing_code)
            print("LLM 생성 코드로 범주형 인코딩 완료")
        except Exception as e:
            print(f"LLM 생성 코드 실행 오류: {e}")
            # 폴백: 기본 자동 처리
            df = apply_basic_categorical_encoding(df, categorical_columns)
    else:
        # 수동 방법 사용
        df = apply_manual_categorical_encoding(df, categorical_columns, method, inputs)
    
    # 인코딩 후 컬럼 정보 업데이트
    final_columns = df.columns.tolist()
    encoding_info = {
        'final_columns': final_columns,
        'column_count_change': len(final_columns) - len(categorical_columns)
    }
    
    print(f"범주형 인코딩 완료: 최종 {len(final_columns)}개 컬럼")
    
    return {
        **inputs,
        "dataframe": df,
        "encoding_info": encoding_info
    }


def generate_categorical_encoding_code_with_llm(df: pd.DataFrame, categorical_columns: List[str],
                                             cate_analysis_text: str, cate_image_paths: List[str],
                                             text_analysis: str) -> str:
    """
    MultiModal LLM을 사용하여 범주형 인코딩 코드를 생성합니다.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=2000
    )
    
    # 범주형 변수 정보 요약
    categorical_summary = []
    for col in categorical_columns:
        unique_count = df[col].nunique()
        value_counts = df[col].value_counts().head(5)
        summary = f"{col}: {unique_count}개 고유값, 상위값={dict(value_counts)}"
        categorical_summary.append(summary)
    
    prompt = f"""
You are a data preprocessing expert. Please write Python code to encode categorical variables based on the following information.

=== Dataset Information ===
- Data size: {df.shape[0]} rows x {df.shape[1]} columns
- Categorical columns: {categorical_columns}
- Dataset head:
{df.head().to_string()}

=== Categorical Variable Information ===
{chr(10).join(categorical_summary)}

=== Categorical Variable Analysis Results ===
{cate_analysis_text}

=== Overall Data Analysis ===
{text_analysis}

=== Requirements ===
1. Use Label Encoding for variables with 10 or fewer unique values
2. Use One-Hot Encoding for variables with many unique values
3. Consider Ordinal Encoding for ordinal categorical variables
4. Code must be executable

Please write code in the following format:
```python
# Categorical variable encoding code
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


def apply_basic_categorical_encoding(df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    """
    기본 범주형 인코딩 방법을 적용합니다.
    """
    for col in categorical_columns:
        unique_count = df[col].nunique()
        print(f"  {col}: {unique_count}개 고유값")
        
        if unique_count <= 10:
            # 고유값이 10개 이하면 Label Encoding
            df[col] = df[col].astype('category').cat.codes
            print(f"    → Label Encoding 적용")
        else:
            # 고유값이 많으면 One-Hot Encoding
            df = pd.get_dummies(df, columns=[col], prefix=col)
            print(f"    → One-Hot Encoding 적용")
    
    return df


def apply_manual_categorical_encoding(df: pd.DataFrame, categorical_columns: List[str], 
                                   method: str, inputs: Dict) -> pd.DataFrame:
    """
    수동 범주형 인코딩 방법을 적용합니다.
    """
    for col in categorical_columns:
        unique_count = df[col].nunique()
        print(f"  {col}: {unique_count}개 고유값")
        
        if method == "label":
            # Label Encoding
            df[col] = df[col].astype('category').cat.codes
            print(f"    → Label Encoding 적용")
        
        elif method == "onehot":
            # One-Hot Encoding
            df = pd.get_dummies(df, columns=[col], prefix=col)
            print(f"    → One-Hot Encoding 적용")
        
        elif method == "ordinal":
            # Ordinal Encoding (순서가 있는 경우)
            # 알파벳 순서로 정렬하여 인코딩
            unique_values = sorted(df[col].unique())
            value_to_code = {val: idx for idx, val in enumerate(unique_values)}
            df[col] = df[col].map(value_to_code)
            print(f"    → Ordinal Encoding 적용")
        
        elif method == "target":
            # Target Encoding (타겟 변수가 있는 경우)
            target_col = inputs.get("target_column")
            if target_col and target_col in df.columns:
                # 타겟 변수의 평균값으로 인코딩
                target_means = df.groupby(col)[target_col].mean()
                df[col] = df[col].map(target_means)
                print(f"    → Target Encoding 적용")
            else:
                print(f"    ⚠️  Target Encoding을 위한 타겟 변수가 없어 Label Encoding으로 대체")
                df[col] = df[col].astype('category').cat.codes
        
        elif method == "frequency":
            # Frequency Encoding (빈도 기반)
            value_counts = df[col].value_counts()
            df[col] = df[col].map(value_counts)
            print(f"    → Frequency Encoding 적용")
    
    return df


# LangGraph 노드로 사용할 수 있는 함수
cate_encoding_agent = RunnableLambda(encode_categorical)