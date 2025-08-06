"""
스케일링 전처리 에이전트
수치형 변수의 스케일을 조정하는 다양한 정규화 방법을 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def scale_features(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    수치형 변수의 스케일을 조정하는 전처리 함수
    
    Args:
        inputs: DataFrame과 스케일링 방법이 포함된 입력 딕셔너리
        
    Returns:
        스케일링된 DataFrame이 포함된 딕셔너리
    """
    df = inputs["dataframe"].copy()
    method = inputs.get("scaling_method", "auto")  # auto, standard, minmax, robust, normalize
    columns = inputs.get("scaling_columns", None)  # 특정 컬럼만 스케일링
    
    # EDA 결과물들 가져오기
    numeric_analysis_text = inputs.get("numeric_analysis_text", "")
    numeric_image_paths = inputs.get("numeric_image_paths", [])
    outlier_analysis_text = inputs.get("outlier_analysis_text", "")
    text_analysis = inputs.get("text_analysis", "")
    
    # 수치형 컬럼 선택
    if columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_columns = [col for col in columns if col in df.columns and df[col].dtype in [np.number]]
    
    if not numeric_columns:
        print("스케일링: 수치형 변수가 없습니다.")
        return {
            **inputs,
            "dataframe": df,
            "scaling_info": {}
        }
    
    print(f"스케일링 시작: {method} 방법")
    print(f"  대상 컬럼: {numeric_columns}")
    
    # MultiModal LLM을 사용한 전처리 코드 생성
    if method == "auto":
        preprocessing_code = generate_scaling_code_with_llm(
            df, numeric_columns, numeric_analysis_text, numeric_image_paths,
            outlier_analysis_text, text_analysis
        )
        
        # 생성된 코드 실행
        try:
            exec(preprocessing_code)
            print("LLM 생성 코드로 스케일링 완료")
        except Exception as e:
            print(f"LLM 생성 코드 실행 오류: {e}")
            # 폴백: 기본 자동 처리
            df = apply_basic_scaling(df, numeric_columns)
    else:
        # 수동 방법 사용
        df = apply_manual_scaling(df, numeric_columns, method, inputs)
    
    # 스케일링 정보 생성
    scaling_info = {}
    for col in numeric_columns:
        original_stats = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
        scaling_info[col] = {
            'original_stats': original_stats,
            'method': method
        }
    
    print(f"스케일링 완료: {len(numeric_columns)}개 컬럼 처리")
    
    return {
        **inputs,
        "dataframe": df,
        "scaling_info": scaling_info
    }


def generate_scaling_code_with_llm(df: pd.DataFrame, numeric_columns: List[str],
                                 numeric_analysis_text: str, numeric_image_paths: List[str],
                                 outlier_analysis_text: str, text_analysis: str) -> str:
    """
    MultiModal LLM을 사용하여 스케일링 코드를 생성합니다.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=2000
    )
    
    # 수치형 변수 정보 요약
    numeric_summary = []
    for col in numeric_columns:
        stats = df[col].describe()
        summary = f"{col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, range=[{stats['min']:.2f}, {stats['max']:.2f}]"
        numeric_summary.append(summary)
    
    prompt = f"""
You are a data preprocessing expert. Please write Python code to scale numeric variables based on the following information.

=== Dataset Information ===
- Data size: {df.shape[0]} rows x {df.shape[1]} columns
- Numeric columns: {numeric_columns}
- Dataset head:
{df.head().to_string()}

=== Numeric Variable Statistics ===
{chr(10).join(numeric_summary)}

=== Numeric Variable Analysis Results ===
{numeric_analysis_text}

=== Outlier Analysis Results ===
{outlier_analysis_text}

=== Overall Data Analysis ===
{text_analysis}

=== Requirements ===
1. Use StandardScaler for variables with large standard deviation
2. Use MinMaxScaler for variables with small standard deviation
3. Consider RobustScaler for variables with many outliers
4. Code must be executable

Please write code in the following format:
```python
# Scaling code
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


def apply_basic_scaling(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """
    기본 스케일링 방법을 적용합니다.
    """
    for col in numeric_columns:
        original_stats = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'range': df[col].max() - df[col].min()
        }
        
        print(f"  {col}: 범위 {original_stats['min']:.2f} ~ {original_stats['max']:.2f}")
        
        # 자동 선택: 분포 특성에 따라 결정
        if original_stats['std'] > 100:
            # 표준편차가 큰 경우 StandardScaler
            df[col] = (df[col] - original_stats['mean']) / original_stats['std']
            print(f"    → StandardScaler 적용")
        else:
            # 표준편차가 작은 경우 MinMaxScaler
            df[col] = (df[col] - original_stats['min']) / original_stats['range']
            print(f"    → MinMaxScaler 적용")
    
    return df


def apply_manual_scaling(df: pd.DataFrame, numeric_columns: List[str], 
                        method: str, inputs: Dict) -> pd.DataFrame:
    """
    수동 스케일링 방법을 적용합니다.
    """
    for col in numeric_columns:
        original_stats = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'range': df[col].max() - df[col].min()
        }
        
        print(f"  {col}: 범위 {original_stats['min']:.2f} ~ {original_stats['max']:.2f}")
        
        if method == "standard":
            # StandardScaler (Z-score normalization)
            df[col] = (df[col] - original_stats['mean']) / original_stats['std']
            print(f"    → StandardScaler 적용")
        
        elif method == "minmax":
            # MinMaxScaler (0-1 정규화)
            df[col] = (df[col] - original_stats['min']) / original_stats['range']
            print(f"    → MinMaxScaler 적용")
        
        elif method == "robust":
            # RobustScaler (중앙값과 IQR 사용)
            median_val = df[col].median()
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR != 0:
                df[col] = (df[col] - median_val) / IQR
            else:
                # IQR이 0인 경우 표준화 사용
                df[col] = (df[col] - original_stats['mean']) / original_stats['std']
                print(f"    ⚠️  IQR이 0이어서 StandardScaler로 대체")
            
            print(f"    → RobustScaler 적용")
        
        elif method == "normalize":
            # L2 정규화 (벡터 정규화)
            l2_norm = np.sqrt((df[col] ** 2).sum())
            if l2_norm != 0:
                df[col] = df[col] / l2_norm
            print(f"    → L2 정규화 적용")
        
        elif method == "log":
            # 로그 변환 (양수 값만)
            if (df[col] > 0).all():
                df[col] = np.log(df[col])
                print(f"    → 로그 변환 적용")
            else:
                # 음수 값이 있는 경우 표준화 사용
                df[col] = (df[col] - original_stats['mean']) / original_stats['std']
                print(f"    ⚠️  음수 값이 있어 StandardScaler로 대체")
    
    return df


# LangGraph 노드로 사용할 수 있는 함수
scaling_agent = RunnableLambda(scale_features)