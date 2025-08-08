"""
범주형 데이터 분석 텍스트 에이전트
unique 값, value_counts, 카테고리 불균형 등을 분석하여 raw한 통계 데이터를 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda


def analyze_categorical_data(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    범주형 데이터의 통계적 분석을 수행합니다.
    
    Args:
        inputs: DataFrame이 포함된 입력 딕셔너리
        
    Returns:
        분석 결과가 포함된 딕셔너리
    """
    print("🔍 [EDA] 범주형 변수 분석 시작...")
    
    df = inputs["dataframe"]
    
    # 범주형 컬럼만 선택
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_columns:
        print("⚠️  [EDA] 범주형 변수가 없습니다.")
        return {
            **inputs,
            "categorical_analysis": {
                "message": "범주형 변수가 없습니다.",
                "statistics": {},
                "columns_analyzed": []
            }
        }
    
    print(f"📊 [EDA] {len(categorical_columns)}개 범주형 변수 분석 중: {categorical_columns}")
    
    # 각 범주형 변수에 대한 통계 분석
    statistics = {}
    
    for col in categorical_columns:
        print(f"   📈 [EDA] {col} 변수 분석 중...")
        
        # 기본 통계량
        value_counts = df[col].value_counts()
        unique_count = df[col].nunique()
        missing_count = df[col].isnull().sum()
        
        statistics[col] = {
            'unique_count': unique_count,
            'missing_count': missing_count,
            'missing_ratio': (missing_count / len(df)) * 100,
            'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
            'most_common_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'most_common_ratio': (value_counts.iloc[0] / len(df)) * 100 if len(value_counts) > 0 else 0,
            'value_counts': value_counts.to_dict(),
            'total_count': len(df)
        }
        
        print(f"   ✅ [EDA] {col} 분석 완료")
    
    print(f"✅ [EDA] 범주형 변수 분석 완료 - 총 {len(categorical_columns)}개 변수")
    
    # 분석 결과 로깅
    print("\n" + "="*50)
    print("📊 [EDA] 범주형 변수 분석 결과")
    print("="*50)
    for col, stats in statistics.items():
        print(f"\n📈 {col}:")
        print(f"   - 고유값: {stats['unique_count']}개")
        print(f"   - 결측값: {stats['missing_count']}개 ({stats['missing_ratio']:.1f}%)")
        print(f"   - 최빈값: {stats['most_common']} ({stats['most_common_count']}개, {stats['most_common_ratio']:.1f}%)")
    
    result = {
        **inputs,
        "categorical_analysis": {
            "statistics": statistics,
            "columns_analyzed": categorical_columns
        }
    }
    
    return result


# LangGraph 노드로 사용할 수 있는 함수
categorical_text_agent = RunnableLambda(analyze_categorical_data)
