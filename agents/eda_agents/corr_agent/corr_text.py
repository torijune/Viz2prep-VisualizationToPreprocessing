"""
변수별 상관관계 분석 텍스트 에이전트
피어슨 상관계수, corr() 함수, 타겟 변수와의 상관관계 등을 분석하여 raw한 상관관계 데이터를 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda


def analyze_correlations(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    변수 간 상관관계를 분석하고 raw한 상관관계 데이터를 제공합니다.
    
    Args:
        inputs: DataFrame이 포함된 입력 딕셔너리
        
    Returns:
        상관관계 분석 결과가 포함된 딕셔너리
    """
    df = inputs["dataframe"]
    
    # 수치형 컬럼만 선택
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 2:
        return {
            **inputs,
            "correlation_analysis": {
                "message": "수치형 변수가 2개 미만이어서 상관관계 분석을 수행할 수 없습니다.",
                "correlation_matrix": {},
                "columns_analyzed": []
            }
        }
    
    # 전체 상관관계 행렬
    correlation_matrix = df[numeric_columns].corr()
    
    # 타겟 변수 식별
    target_column = None
    for col in ['Survived', 'target', 'label', 'class', 'y']:
        if col in numeric_columns:
            target_column = col
            break
    
    # 타겟 변수와의 상관관계
    target_correlations = {}
    if target_column:
        target_correlations = correlation_matrix[target_column].to_dict()
    
    # 모든 변수 쌍의 상관관계
    all_correlations = []
    for i, col1 in enumerate(numeric_columns):
        for j, col2 in enumerate(numeric_columns[i+1:], i+1):
            corr_value = correlation_matrix.loc[col1, col2]
            all_correlations.append({
                'var1': col1,
                'var2': col2,
                'correlation': corr_value
            })
    
    # 상관관계 통계
    correlation_values = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
    correlation_stats = {
        'mean': float(np.mean(correlation_values)),
        'std': float(np.std(correlation_values)),
        'min': float(np.min(correlation_values)),
        'max': float(np.max(correlation_values)),
        'positive_count': int((correlation_values > 0).sum()),
        'negative_count': int((correlation_values < 0).sum()),
        'zero_count': int((correlation_values == 0).sum()),
        'total_pairs': len(correlation_values)
    }
    
    print("상관관계 분석 완료")
    
    # 분석 결과 로깅
    print("\n" + "="*50)
    print("📊 [EDA] 상관관계 분석 결과")
    print("="*50)
    print(f"📈 분석된 변수: {len(numeric_columns)}개")
    if target_column:
        print(f"🎯 타겟 변수: {target_column}")
        # 타겟 변수와의 상관관계 상위 5개
        target_corr_sorted = sorted(target_correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
        print(f"   - 타겟 변수와의 상관관계 (상위 5개):")
        for var, corr in target_corr_sorted:
            if var != target_column:
                print(f"     * {var}: {corr:.4f}")
    
    print(f"📊 상관관계 통계:")
    print(f"   - 평균: {correlation_stats['mean']:.4f}")
    print(f"   - 표준편차: {correlation_stats['std']:.4f}")
    print(f"   - 범위: {correlation_stats['min']:.4f} ~ {correlation_stats['max']:.4f}")
    print(f"   - 양의 상관관계: {correlation_stats['positive_count']}개")
    print(f"   - 음의 상관관계: {correlation_stats['negative_count']}개")
    
    # 강한 상관관계 (|r| >= 0.5) 출력
    strong_correlations = [corr for corr in all_correlations if abs(corr['correlation']) >= 0.5]
    if strong_correlations:
        print(f"🔍 강한 상관관계 (|r| >= 0.5): {len(strong_correlations)}개")
        for corr in strong_correlations[:5]:  # 상위 5개만 출력
            print(f"   - {corr['var1']} ↔ {corr['var2']}: {corr['correlation']:.4f}")
    
    result = {
        **inputs,
        "correlation_analysis": {
            "correlation_matrix": correlation_matrix.to_dict(),
            "target_column": target_column,
            "target_correlations": target_correlations,
            "all_correlations": all_correlations,
            "correlation_stats": correlation_stats,
            "columns_analyzed": numeric_columns
        }
    }
    
    return result


# LangGraph 노드로 사용할 수 있는 함수
correlation_text_agent = RunnableLambda(analyze_correlations)
