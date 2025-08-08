"""
이상치 데이터 분석 텍스트 에이전트
사분위수 기반 방법, Z-Score 등을 사용하여 이상치를 분석하고 raw한 이상치 데이터를 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda


def analyze_outliers(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    이상치를 분석하고 raw한 이상치 데이터를 제공합니다.
    
    Args:
        inputs: DataFrame이 포함된 입력 딕셔너리
        
    Returns:
        이상치 분석 결과가 포함된 딕셔너리
    """
    df = inputs["dataframe"]
    
    # 수치형 컬럼만 선택
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_columns:
        return {
            **inputs,
            "outlier_analysis": {
                "message": "수치형 데이터가 없어서 이상치 분석을 수행할 수 없습니다.",
                "iqr_outliers": {},
                "zscore_outliers": {},
                "columns_analyzed": []
            }
        }
    
    # IQR 기반 이상치 분석
    iqr_outliers = {}
    for col in numeric_columns:
        print(f"🔍 [EDA] {col} 이상치 분석 중...")
        
        # 기본 통계량
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 이상치 탐지
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(df)) * 100
        
        # 디버깅 정보 출력
        print(f"   📊 {col} 통계:")
        print(f"     - Q1: {Q1:.4f}, Q3: {Q3:.4f}, IQR: {IQR:.4f}")
        print(f"     - 하한: {lower_bound:.4f}, 상한: {upper_bound:.4f}")
        print(f"     - 데이터 범위: {df[col].min():.4f} ~ {df[col].max():.4f}")
        print(f"     - 이상치 개수: {outlier_count}개 ({outlier_percentage:.1f}%)")
        
        iqr_outliers[col] = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': outlier_count,
            'outlier_percentage': outlier_percentage,
            'data_min': float(df[col].min()),
            'data_max': float(df[col].max())
        }
        
        if outlier_count > 0:
            iqr_outliers[col].update({
                'min_outlier': float(outliers[col].min()),
                'max_outlier': float(outliers[col].max())
            })
    
    # Z-Score 기반 이상치 분석
    zscore_outliers = {}
    for col in numeric_columns:
        print(f"🔍 [EDA] {col} Z-Score 이상치 분석 중...")
        
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers = df[z_scores > 3]  # Z-Score > 3을 이상치로 정의
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(df)) * 100
        
        # 디버깅 정보 출력
        print(f"   📊 {col} Z-Score 통계:")
        print(f"     - 평균: {df[col].mean():.4f}, 표준편차: {df[col].std():.4f}")
        print(f"     - 최대 Z-Score: {z_scores.max():.4f}")
        print(f"     - 이상치 개수: {outlier_count}개 ({outlier_percentage:.1f}%)")
        
        zscore_outliers[col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'outlier_count': outlier_count,
            'outlier_percentage': outlier_percentage,
            'max_z_score': float(z_scores.max())
        }
        
        if outlier_count > 0:
            zscore_outliers[col].update({
                'min_outlier': float(outliers[col].min()),
                'max_outlier': float(outliers[col].max())
            })
    
    # 전체 이상치 통계
    total_outliers_iqr = sum(iqr_outliers[col]['outlier_count'] for col in numeric_columns)
    total_outliers_zscore = sum(zscore_outliers[col]['outlier_count'] for col in numeric_columns)
    
    outlier_stats = {
        'total_outliers_iqr': total_outliers_iqr,
        'total_outliers_zscore': total_outliers_zscore,
        'columns_analyzed': len(numeric_columns)
    }
    
    print("이상치 분석 완료")
    
    # 분석 결과 로깅
    print("\n" + "="*50)
    print("📊 [EDA] 이상치 분석 결과")
    print("="*50)
    print(f"📈 분석된 변수: {len(numeric_columns)}개")
    print(f"🔍 전체 이상치 (IQR): {total_outliers_iqr}개")
    print(f"🔍 전체 이상치 (Z-Score): {total_outliers_zscore}개")
    
    for col in numeric_columns:
        print(f"\n📈 {col}:")
        iqr_count = iqr_outliers[col]['outlier_count']
        zscore_count = zscore_outliers[col]['outlier_count']
        print(f"   - IQR 이상치: {iqr_count}개 ({iqr_outliers[col]['outlier_percentage']:.1f}%)")
        print(f"   - Z-Score 이상치: {zscore_count}개 ({zscore_outliers[col]['outlier_percentage']:.1f}%)")
        
        # 데이터 분포 특성 출력
        data_min = iqr_outliers[col]['data_min']
        data_max = iqr_outliers[col]['data_max']
        print(f"   - 데이터 범위: {data_min:.4f} ~ {data_max:.4f}")
        
        if iqr_count > 0:
            print(f"   - IQR 범위: {iqr_outliers[col]['lower_bound']:.4f} ~ {iqr_outliers[col]['upper_bound']:.4f}")
        if zscore_count > 0:
            print(f"   - 최대 Z-Score: {zscore_outliers[col]['max_z_score']:.4f}")
        
        # 이상치가 0개인 경우 원인 분석
        if iqr_count == 0 and zscore_count == 0:
            print(f"   ⚠️  이상치가 0개인 이유:")
            print(f"     - 데이터 분포가 매우 균등하거나")
            print(f"     - 데이터 범위가 좁거나")
            print(f"     - 분포가 매우 치우쳐 있어서 기준값이 데이터 범위를 벗어남")
    
    result = {
        **inputs,
        "outlier_analysis": {
            "iqr_outliers": iqr_outliers,
            "zscore_outliers": zscore_outliers,
            "outlier_stats": outlier_stats,
            "columns_analyzed": numeric_columns
        }
    }
    
    return result


# LangGraph 노드로 사용할 수 있는 함수
outlier_text_agent = RunnableLambda(analyze_outliers)
