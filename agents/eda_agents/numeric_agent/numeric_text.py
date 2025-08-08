"""
연속형 데이터 통계치 분석 텍스트 에이전트
Min, Max, Mean, 분포 왜도, 첨도 등을 분석하여 raw한 통계 데이터를 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda


def analyze_numeric_statistics(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    연속형 데이터의 통계적 분석을 수행합니다.
    
    Args:
        inputs: DataFrame이 포함된 입력 딕셔너리
        
    Returns:
        분석 결과가 포함된 딕셔너리
    """
    print("🔍 [EDA] 수치형 변수 통계 분석 시작...")
    
    df = inputs["dataframe"]
    
    # 수치형 컬럼만 선택
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_columns:
        print("⚠️  [EDA] 수치형 변수가 없습니다.")
        
        # 숫자로 변환 가능한 컬럼들 찾기
        potential_numeric_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # 숫자로 변환 가능한지 테스트
                try:
                    pd.to_numeric(df[col], errors='raise')
                    potential_numeric_columns.append(col)
                except (ValueError, TypeError):
                    pass
        
        if potential_numeric_columns:
            print(f"💡 [EDA] 숫자로 변환 가능한 컬럼 발견: {potential_numeric_columns}")
            print("   → 이 컬럼들을 수치형으로 변환하면 분석이 가능합니다.")
        
        print(f"📊 [EDA] 전체 컬럼 타입:")
        for col in df.columns:
            print(f"   - {col}: {df[col].dtype}")
        
        return {
            **inputs,
            "numeric_analysis": {
                "message": "수치형 변수가 없습니다.",
                "statistics": {},
                "columns_analyzed": [],
                "potential_numeric_columns": potential_numeric_columns
            }
        }
    
    print(f"📊 [EDA] {len(numeric_columns)}개 수치형 변수 분석 중: {numeric_columns}")
    
    # 각 수치형 변수에 대한 통계 분석
    statistics = {}
    
    for col in numeric_columns:
        print(f"   📈 [EDA] {col} 변수 분석 중...")
        
        # 기본 통계량
        stats = df[col].describe()
        statistics[col] = {
            'count': stats['count'],
            'mean': stats['mean'],
            'std': stats['std'],
            'min': stats['min'],
            '25%': stats['25%'],
            'median': stats['50%'],
            '75%': stats['75%'],
            'max': stats['max'],
            'skewness': df[col].skew(),
            'kurtosis': df[col].kurtosis(),
            'missing_count': df[col].isnull().sum(),
            'missing_ratio': (df[col].isnull().sum() / len(df)) * 100
        }
        
        # IQR 기반 이상치 정보 (raw 데이터)
        Q1 = stats['25%']
        Q3 = stats['75%']
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        
        statistics[col].update({
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': len(outliers),
            'outlier_ratio': (len(outliers) / len(df)) * 100
        })
        
        print(f"   ✅ [EDA] {col} 분석 완료")
    
    print(f"✅ [EDA] 수치형 변수 분석 완료 - 총 {len(numeric_columns)}개 변수")
    
    # 분석 결과 로깅
    print("\n" + "="*50)
    print("📊 [EDA] 수치형 변수 분석 결과")
    print("="*50)
    for col, stats in statistics.items():
        print(f"\n📈 {col}:")
        print(f"   - 기본 통계: 평균={stats['mean']:.4f}, 표준편차={stats['std']:.4f}")
        print(f"   - 분포: 왜도={stats['skewness']:.4f}, 첨도={stats['kurtosis']:.4f}")
        print(f"   - 결측값: {stats['missing_count']}개 ({stats['missing_ratio']:.1f}%)")
        print(f"   - 이상치: {stats['outlier_count']}개 ({stats['outlier_ratio']:.1f}%)")
    
    result = {
        **inputs,
        "numeric_analysis": {
            "statistics": statistics,
            "columns_analyzed": numeric_columns
        }
    }
    
    return result


# LangGraph 노드로 사용할 수 있는 함수
numeric_text_agent = RunnableLambda(analyze_numeric_statistics)
