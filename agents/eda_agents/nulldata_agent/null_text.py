"""
결측값 텍스트 분석 에이전트
결측값 분포를 텍스트로 분석하여 LLM이 이해하기 쉬운 형태로 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda


def analyze_null_data_text(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    결측값 분포를 텍스트로 분석하는 함수
    
    Args:
        inputs: DataFrame이 포함된 입력 딕셔너리
        
    Returns:
        결측값 분석 텍스트가 포함된 딕셔너리
    """
    df = inputs["dataframe"]
    
    # 결측값 분석 결과를 저장할 리스트
    analysis_parts = []
    
    # 1. 전체 결측값 현황
    analysis_parts.append("=== 결측값 전체 현황 ===")
    total_missing = df.isnull().sum().sum()
    total_cells = len(df) * len(df.columns)
    missing_ratio = (total_missing / total_cells) * 100
    
    analysis_parts.append(f"전체 결측값: {total_missing}개")
    analysis_parts.append(f"전체 데이터 셀: {total_cells}개")
    analysis_parts.append(f"결측값 비율: {missing_ratio:.2f}%")
    analysis_parts.append("")
    
    # 2. 컬럼별 결측값 분석
    analysis_parts.append("=== 컬럼별 결측값 분석 ===")
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    # 결측값이 있는 컬럼만 필터링
    columns_with_missing = missing_counts[missing_counts > 0]
    
    if len(columns_with_missing) > 0:
        analysis_parts.append("결측값이 있는 컬럼:")
        for col in columns_with_missing.index:
            count = missing_counts[col]
            percentage = missing_percentages[col]
            analysis_parts.append(f"  {col}: {count}개 ({percentage:.1f}%)")
    else:
        analysis_parts.append("결측값이 있는 컬럼이 없습니다.")
    
    analysis_parts.append("")
    
    # 3. 결측값 패턴 분석
    if len(columns_with_missing) > 1:
        analysis_parts.append("=== 결측값 패턴 분석 ===")
        
        # 결측값이 있는 컬럼들 간의 상관관계
        null_cols = columns_with_missing.index.tolist()
        null_corr = df[null_cols].isnull().corr()
        
        # 높은 상관관계를 보이는 쌍 찾기
        high_corr_pairs = []
        for i in range(len(null_cols)):
            for j in range(i+1, len(null_cols)):
                corr_val = null_corr.iloc[i, j]
                if abs(corr_val) > 0.5:
                    high_corr_pairs.append((null_cols[i], null_cols[j], corr_val))
        
        if high_corr_pairs:
            analysis_parts.append("높은 상관관계를 보이는 결측값 패턴:")
            for var1, var2, corr_val in high_corr_pairs:
                analysis_parts.append(f"  {var1} ↔ {var2}: {corr_val:.3f}")
        else:
            analysis_parts.append("결측값 간 특별한 패턴이 없습니다.")
        
        analysis_parts.append("")
    
    # 4. 결측값 심각도 평가
    analysis_parts.append("=== 결측값 심각도 평가 ===")
    
    # 심각한 결측값 (50% 이상)
    severe_missing = missing_percentages[missing_percentages > 50]
    if len(severe_missing) > 0:
        analysis_parts.append("⚠️  심각한 결측값 (>50%):")
        for col in severe_missing.index:
            analysis_parts.append(f"  - {col}: {severe_missing[col]:.1f}%")
    else:
        analysis_parts.append("✅ 심각한 결측값이 없습니다.")
    
    # 중간 수준 결측값 (10-50%)
    moderate_missing = missing_percentages[(missing_percentages > 10) & (missing_percentages <= 50)]
    if len(moderate_missing) > 0:
        analysis_parts.append("\n⚠️  중간 수준 결측값 (10-50%):")
        for col in moderate_missing.index:
            analysis_parts.append(f"  - {col}: {moderate_missing[col]:.1f}%")
    
    # 경미한 결측값 (1-10%)
    minor_missing = missing_percentages[(missing_percentages > 0) & (missing_percentages <= 10)]
    if len(minor_missing) > 0:
        analysis_parts.append("\nℹ️  경미한 결측값 (1-10%):")
        for col in minor_missing.index:
            analysis_parts.append(f"  - {col}: {minor_missing[col]:.1f}%")
    
    analysis_parts.append("")
    
    # 5. 결측값 처리 권장사항
    analysis_parts.append("=== 결측값 처리 권장사항 ===")
    
    for col in df.columns:
        missing_pct = missing_percentages[col]
        if missing_pct > 0:
            if missing_pct > 50:
                analysis_parts.append(f"  {col}: 결측값이 많으므로 컬럼 삭제 고려")
            elif missing_pct > 10:
                if df[col].dtype in ['object', 'category']:
                    analysis_parts.append(f"  {col}: 최빈값으로 대체 권장")
                else:
                    analysis_parts.append(f"  {col}: 평균/중앙값으로 대체 권장")
            else:
                if df[col].dtype in ['object', 'category']:
                    analysis_parts.append(f"  {col}: 최빈값으로 대체")
                else:
                    analysis_parts.append(f"  {col}: 평균/중앙값으로 대체")
    
    # 6. 결측값 유형 분석
    analysis_parts.append("\n=== 결측값 유형 분석 ===")
    
    # MCAR (Missing Completely At Random) - 완전 무작위
    # MAR (Missing At Random) - 무작위
    # MNAR (Missing Not At Random) - 비무작위
    
    if len(columns_with_missing) > 0:
        # 간단한 패턴 분석
        has_pattern = False
        if len(columns_with_missing) > 1:
            null_corr = df[columns_with_missing.index].isnull().corr()
            if (null_corr > 0.3).sum().sum() > len(columns_with_missing):
                has_pattern = True
        
        if has_pattern:
            analysis_parts.append("결측값 유형: MAR (Missing At Random) - 변수 간 상관관계가 있음")
        else:
            analysis_parts.append("결측값 유형: MCAR (Missing Completely At Random) - 완전 무작위")
    else:
        analysis_parts.append("결측값이 없어 유형 분석이 불필요합니다.")
    
    # 텍스트 분석 결과 생성
    null_analysis_text = "\n".join(analysis_parts)
    
    print("결측값 텍스트 분석 완료")
    
    return {
        **inputs,
        "dataframe": df,
        "null_analysis_text": null_analysis_text
    }


# LangGraph 노드로 사용할 수 있는 함수
null_text_agent = RunnableLambda(analyze_null_data_text)
