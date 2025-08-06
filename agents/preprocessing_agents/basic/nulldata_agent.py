"""
결측값 처리 전처리 에이전트
데이터의 결측값을 다양한 방법으로 처리합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def handle_missing_data(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    결측값을 처리하는 전처리 함수
    
    Args:
        inputs: DataFrame과 처리 방법이 포함된 입력 딕셔너리
        
    Returns:
        결측값이 처리된 DataFrame이 포함된 딕셔너리
    """
    df = inputs["dataframe"].copy()
    method = inputs.get("missing_method", "auto")  # auto, drop, fill, impute
    
    # EDA 결과물들 가져오기
    null_analysis_text = inputs.get("null_analysis_text", "")
    null_image_paths = inputs.get("null_image_paths", [])
    corr_analysis_text = inputs.get("corr_analysis_text", "")
    text_analysis = inputs.get("text_analysis", "")
    
    # 결측값 현황 분석
    missing_info = {}
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        missing_info[col] = {
            'count': missing_count,
            'percentage': missing_pct
        }
    
    print(f"결측값 처리 시작: {method} 방법 사용")
    
    # MultiModal LLM을 사용한 전처리 코드 생성
    if method == "auto":
        preprocessing_code = generate_missing_data_code_with_llm(
            df, missing_info, null_analysis_text, null_image_paths, 
            corr_analysis_text, text_analysis
        )
        
        # 생성된 코드 실행
        try:
            exec(preprocessing_code)
            print("LLM 생성 코드로 결측값 처리 완료")
        except Exception as e:
            print(f"LLM 생성 코드 실행 오류: {e}")
            # 폴백: 기본 자동 처리
            df = apply_basic_missing_data_handling(df, missing_info)
    else:
        # 수동 방법 사용
        df = apply_manual_missing_data_handling(df, method, inputs)
    
    # 처리 후 결측값 확인
    remaining_missing = df.isnull().sum().sum()
    print(f"결측값 처리 완료: 남은 결측값 {remaining_missing}개")
    
    return {
        **inputs,
        "dataframe": df,
        "missing_info": missing_info,
        "remaining_missing": remaining_missing
    }


def generate_missing_data_code_with_llm(df: pd.DataFrame, missing_info: Dict, 
                                      null_analysis_text: str, null_image_paths: List[str],
                                      corr_analysis_text: str, text_analysis: str) -> str:
    """
    MultiModal LLM을 사용하여 결측값 처리 코드를 생성합니다.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=2000
    )
    
    # 결측값 정보 요약
    missing_summary = []
    for col, info in missing_info.items():
        if info['count'] > 0:
            missing_summary.append(f"{col}: {info['count']}개 ({info['percentage']:.1f}%)")
    
    prompt = f"""
You are a data preprocessing expert. Please write Python code to handle missing values based on the following information.

=== Dataset Information ===
- Data size: {df.shape[0]} rows x {df.shape[1]} columns
- Columns: {list(df.columns)}
- Data types: {dict(df.dtypes)}
- Dataset head:
{df.head().to_string()}

=== Missing Value Status ===
{chr(10).join(missing_summary)}

=== Missing Value Analysis Results ===
{null_analysis_text}

=== Correlation Analysis Results ===
{corr_analysis_text}

=== Overall Data Analysis ===
{text_analysis}

=== Requirements ===
1. Drop columns with more than 50% missing values
2. Fill categorical variables with mode
3. Fill numerical variables with median
4. Consider missing value patterns if present
5. Code must be executable

Please write code in the following format:
```python
# Missing value handling code
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


def apply_basic_missing_data_handling(df: pd.DataFrame, missing_info: Dict) -> pd.DataFrame:
    """
    기본 결측값 처리 방법을 적용합니다.
    """
    for col in df.columns:
        missing_pct = missing_info[col]['percentage']
        
        if missing_pct > 50:
            # 결측값이 50% 이상인 경우 컬럼 삭제
            df = df.drop(columns=[col])
            print(f"  {col}: 결측값 {missing_pct:.1f}% → 컬럼 삭제")
        elif missing_pct > 0:
            # 결측값이 있는 경우 데이터 타입에 따라 처리
            if df[col].dtype in ['object', 'category']:
                # 범주형 변수는 최빈값으로 대체
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_val)
                print(f"  {col}: 최빈값으로 대체 ({mode_val})")
            else:
                # 수치형 변수는 중앙값으로 대체
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"  {col}: 중앙값으로 대체 ({median_val:.2f})")
    
    return df


def apply_manual_missing_data_handling(df: pd.DataFrame, method: str, inputs: Dict) -> pd.DataFrame:
    """
    수동 결측값 처리 방법을 적용합니다.
    """
    if method == "drop":
        # 결측값이 있는 행 삭제
        original_len = len(df)
        df = df.dropna()
        dropped_count = original_len - len(df)
        print(f"  결측값이 있는 행 {dropped_count}개 삭제")
    
    elif method == "fill":
        # 사용자 지정 값으로 대체
        fill_value = inputs.get("fill_value", 0)
        df = df.fillna(fill_value)
        print(f"  모든 결측값을 {fill_value}로 대체")
    
    elif method == "impute":
        # 고급 대체 방법 (평균, 중앙값, 최빈값)
        impute_method = inputs.get("impute_method", "median")
        
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if impute_method == "mean" and df[col].dtype in [np.number]:
                    df[col] = df[col].fillna(df[col].mean())
                elif impute_method == "median" and df[col].dtype in [np.number]:
                    df[col] = df[col].fillna(df[col].median())
                elif impute_method == "mode":
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                    df[col] = df[col].fillna(mode_val)
                elif impute_method == "forward":
                    df[col] = df[col].fillna(method='ffill')
                elif impute_method == "backward":
                    df[col] = df[col].fillna(method='bfill')
        
        print(f"  {impute_method} 방법으로 결측값 대체")
    
    return df


# LangGraph 노드로 사용할 수 있는 함수
nulldata_agent = RunnableLambda(handle_missing_data)