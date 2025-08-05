"""
결측값 처리 전처리 에이전트
데이터의 결측값을 다양한 방법으로 처리합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableLambda


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
    
    if method == "auto":
        # 자동 처리: 컬럼별 특성에 따라 적절한 방법 선택
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
    
    elif method == "drop":
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
    
    # 처리 후 결측값 확인
    remaining_missing = df.isnull().sum().sum()
    print(f"결측값 처리 완료: 남은 결측값 {remaining_missing}개")
    
    return {
        **inputs,
        "dataframe": df,
        "missing_info": missing_info,
        "remaining_missing": remaining_missing
    }


# LangGraph 노드로 사용할 수 있는 함수
nulldata_agent = RunnableLambda(handle_missing_data)