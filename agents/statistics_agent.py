"""
통계 에이전트
DataFrame의 통계 정보를 분석하고 요약합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda


def generate_statistics(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    DataFrame의 통계 정보를 생성합니다.
    
    Args:
        inputs: DataFrame이 포함된 입력 딕셔너리
        
    Returns:
        통계 요약 텍스트가 포함된 딕셔너리
    """
    df = inputs["dataframe"]
    
    # 기본 통계 정보 생성
    stats_summary = []
    stats_summary.append("=== 데이터셋 기본 정보 ===")
    stats_summary.append(f"데이터셋 크기: {df.shape[0]} 행 x {df.shape[1]} 열")
    stats_summary.append(f"메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 데이터 타입 정보
    stats_summary.append("\n=== 데이터 타입 정보 ===")
    dtype_info = df.dtypes.value_counts()
    for dtype, count in dtype_info.items():
        stats_summary.append(f"{dtype}: {count}개 컬럼")
    
    # 결측값 정보
    stats_summary.append("\n=== 결측값 정보 ===")
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    for col in df.columns:
        if missing_counts[col] > 0:
            stats_summary.append(f"{col}: {missing_counts[col]}개 ({missing_percentages[col]:.2f}%)")
        else:
            stats_summary.append(f"{col}: 결측값 없음")
    
    # 수치형 컬럼 통계
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats_summary.append("\n=== 수치형 컬럼 통계 ===")
        desc_stats = df[numeric_cols].describe()
        stats_summary.append(desc_stats.to_string())
    
    # 범주형 컬럼 통계
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        stats_summary.append("\n=== 범주형 컬럼 통계 ===")
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            stats_summary.append(f"\n{col} (고유값: {df[col].nunique()}개):")
            stats_summary.append(value_counts.head(10).to_string())
    
    statistics_text = "\n".join(stats_summary)
    print("통계 정보 생성 완료")
    print("statistics_text : ", statistics_text)
    
    return {
        "dataframe": df,
        "statistics_text": statistics_text
    }


# LangGraph 노드로 사용할 수 있는 함수
statistics_agent = RunnableLambda(generate_statistics) 