"""
범주형 변수 인코딩 전처리 에이전트
범주형 변수를 수치형으로 변환하는 다양한 인코딩 방법을 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableLambda


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
    
    encoding_info = {}
    
    for col in categorical_columns:
        unique_count = df[col].nunique()
        encoding_info[col] = {
            'unique_count': unique_count,
            'method': method,
            'original_dtype': str(df[col].dtype)
        }
        
        print(f"  {col}: {unique_count}개 고유값")
        
        if method == "auto":
            # 자동 선택: 고유값 개수에 따라 결정
            if unique_count <= 10:
                # 고유값이 10개 이하면 Label Encoding
                df[col] = df[col].astype('category').cat.codes
                encoding_info[col]['method'] = 'label'
                print(f"    → Label Encoding 적용")
            else:
                # 고유값이 많으면 One-Hot Encoding
                df = pd.get_dummies(df, columns=[col], prefix=col)
                encoding_info[col]['method'] = 'onehot'
                print(f"    → One-Hot Encoding 적용")
        
        elif method == "label":
            # Label Encoding
            df[col] = df[col].astype('category').cat.codes
            encoding_info[col]['method'] = 'label'
            print(f"    → Label Encoding 적용")
        
        elif method == "onehot":
            # One-Hot Encoding
            df = pd.get_dummies(df, columns=[col], prefix=col)
            encoding_info[col]['method'] = 'onehot'
            print(f"    → One-Hot Encoding 적용")
        
        elif method == "ordinal":
            # Ordinal Encoding (순서가 있는 경우)
            # 알파벳 순서로 정렬하여 인코딩
            unique_values = sorted(df[col].unique())
            value_to_code = {val: idx for idx, val in enumerate(unique_values)}
            df[col] = df[col].map(value_to_code)
            encoding_info[col]['method'] = 'ordinal'
            encoding_info[col]['value_mapping'] = value_to_code
            print(f"    → Ordinal Encoding 적용")
        
        elif method == "target":
            # Target Encoding (타겟 변수가 있는 경우)
            target_col = inputs.get("target_column")
            if target_col and target_col in df.columns:
                # 타겟 변수의 평균값으로 인코딩
                target_means = df.groupby(col)[target_col].mean()
                df[col] = df[col].map(target_means)
                encoding_info[col]['method'] = 'target'
                encoding_info[col]['target_means'] = target_means.to_dict()
                print(f"    → Target Encoding 적용")
            else:
                print(f"    ⚠️  Target Encoding을 위한 타겟 변수가 없어 Label Encoding으로 대체")
                df[col] = df[col].astype('category').cat.codes
                encoding_info[col]['method'] = 'label'
        
        elif method == "frequency":
            # Frequency Encoding (빈도 기반)
            value_counts = df[col].value_counts()
            df[col] = df[col].map(value_counts)
            encoding_info[col]['method'] = 'frequency'
            print(f"    → Frequency Encoding 적용")
    
    # 인코딩 후 컬럼 정보 업데이트
    final_columns = df.columns.tolist()
    encoding_info['final_columns'] = final_columns
    encoding_info['column_count_change'] = len(final_columns) - len(categorical_columns)
    
    print(f"범주형 인코딩 완료: 최종 {len(final_columns)}개 컬럼")
    
    return {
        **inputs,
        "dataframe": df,
        "encoding_info": encoding_info
    }


# LangGraph 노드로 사용할 수 있는 함수
cate_encoding_agent = RunnableLambda(encode_categorical)