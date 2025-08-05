"""
스케일링 전처리 에이전트
수치형 변수의 스케일을 조정하는 다양한 정규화 방법을 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableLambda


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
    
    scaling_info = {}
    
    for col in numeric_columns:
        original_stats = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'range': df[col].max() - df[col].min()
        }
        
        scaling_info[col] = {
            'original_stats': original_stats,
            'method': method
        }
        
        print(f"  {col}: 범위 {original_stats['min']:.2f} ~ {original_stats['max']:.2f}")
        
        if method == "auto":
            # 자동 선택: 분포 특성에 따라 결정
            if original_stats['std'] > 100:
                # 표준편차가 큰 경우 StandardScaler
                df[col] = (df[col] - original_stats['mean']) / original_stats['std']
                scaling_info[col]['method'] = 'standard'
                print(f"    → StandardScaler 적용")
            else:
                # 표준편차가 작은 경우 MinMaxScaler
                df[col] = (df[col] - original_stats['min']) / original_stats['range']
                scaling_info[col]['method'] = 'minmax'
                print(f"    → MinMaxScaler 적용")
        
        elif method == "standard":
            # StandardScaler (Z-score normalization)
            df[col] = (df[col] - original_stats['mean']) / original_stats['std']
            scaling_info[col]['method'] = 'standard'
            print(f"    → StandardScaler 적용")
        
        elif method == "minmax":
            # MinMaxScaler (0-1 정규화)
            df[col] = (df[col] - original_stats['min']) / original_stats['range']
            scaling_info[col]['method'] = 'minmax'
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
            
            scaling_info[col]['method'] = 'robust'
            print(f"    → RobustScaler 적용")
        
        elif method == "normalize":
            # L2 정규화 (벡터 정규화)
            l2_norm = np.sqrt((df[col] ** 2).sum())
            if l2_norm != 0:
                df[col] = df[col] / l2_norm
            scaling_info[col]['method'] = 'normalize'
            print(f"    → L2 정규화 적용")
        
        elif method == "log":
            # 로그 변환 (양수 값만)
            if (df[col] > 0).all():
                df[col] = np.log(df[col])
                scaling_info[col]['method'] = 'log'
                print(f"    → 로그 변환 적용")
            else:
                # 음수 값이 있는 경우 표준화 사용
                df[col] = (df[col] - original_stats['mean']) / original_stats['std']
                scaling_info[col]['method'] = 'standard'
                print(f"    ⚠️  음수 값이 있어 StandardScaler로 대체")
        
        # 스케일링 후 통계 확인
        scaled_stats = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
        scaling_info[col]['scaled_stats'] = scaled_stats
    
    print(f"스케일링 완료: {len(numeric_columns)}개 컬럼 처리")
    
    return {
        **inputs,
        "dataframe": df,
        "scaling_info": scaling_info
    }


# LangGraph 노드로 사용할 수 있는 함수
scaling_agent = RunnableLambda(scale_features)