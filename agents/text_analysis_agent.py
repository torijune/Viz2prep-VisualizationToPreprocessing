"""
텍스트 기반 통계 분석 에이전트
LLM이 데이터 구조와 특징을 빠르게 이해할 수 있도록 상세한 텍스트 요약을 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda


def analyze_data_text(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    데이터의 구조와 특징을 분석하여 LLM이 이해하기 쉬운 텍스트 요약을 생성합니다.
    
    Args:
        inputs: DataFrame이 포함된 입력 딕셔너리
        
    Returns:
        텍스트 분석 결과가 포함된 딕셔너리
    """
    df = inputs["dataframe"]
    
    # 텍스트 분석 결과를 저장할 리스트
    analysis_parts = []
    
    # 1. 데이터 개요
    analysis_parts.append("=== 데이터 개요 ===")
    analysis_parts.append(f"데이터셋 크기: {df.shape[0]} 행 x {df.shape[1]} 열")
    analysis_parts.append(f"메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    analysis_parts.append("")
    
    # 2. 컬럼 정보 (df.info() 기반)
    analysis_parts.append("=== 컬럼 정보 ===")
    analysis_parts.append(f"총 컬럼 수: {len(df.columns)}")
    
    # 데이터 타입별 분류
    dtype_counts = df.dtypes.value_counts()
    analysis_parts.append("데이터 타입 분포:")
    for dtype, count in dtype_counts.items():
        analysis_parts.append(f"  {dtype}: {count}개 컬럼")
    
    # 컬럼별 상세 정보
    analysis_parts.append("\n컬럼별 상세 정보:")
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        unique_count = df[col].nunique()
        
        analysis_parts.append(f"  {col}:")
        analysis_parts.append(f"    - 데이터 타입: {dtype}")
        analysis_parts.append(f"    - 결측값: {null_count}개 ({null_pct:.1f}%)")
        analysis_parts.append(f"    - 고유값: {unique_count}개")
        
        # 범주형 변수의 경우 상위 값들 표시
        if dtype == 'object' or unique_count < 20:
            top_values = df[col].value_counts().head(5)
            analysis_parts.append(f"    - 상위 값들: {dict(top_values)}")
        
        analysis_parts.append("")
    
    # 3. 기초 통계량 (df.describe())
    analysis_parts.append("=== 기초 통계량 ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        desc_stats = df[numeric_cols].describe()
        analysis_parts.append("수치형 변수 통계:")
        analysis_parts.append(desc_stats.to_string())
        analysis_parts.append("")
    else:
        analysis_parts.append("수치형 변수가 없습니다.")
        analysis_parts.append("")
    
    # 4. 결측치 정보
    analysis_parts.append("=== 결측치 정보 ===")
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    analysis_parts.append("컬럼별 결측치 현황:")
    for col in df.columns:
        if missing_counts[col] > 0:
            analysis_parts.append(f"  {col}: {missing_counts[col]}개 ({missing_percentages[col]:.1f}%)")
        else:
            analysis_parts.append(f"  {col}: 결측값 없음")
    
    total_missing = missing_counts.sum()
    total_cells = len(df) * len(df.columns)
    missing_ratio = (total_missing / total_cells) * 100
    
    analysis_parts.append(f"\n전체 결측치: {total_missing}개 ({missing_ratio:.1f}%)")
    analysis_parts.append("")
    
    # 5. 고유값 및 범주형 분포
    analysis_parts.append("=== 고유값 및 범주형 분포 ===")
    analysis_parts.append("컬럼별 고유값 개수:")
    for col in df.columns:
        unique_count = df[col].nunique()
        analysis_parts.append(f"  {col}: {unique_count}개")
    
    # 범주형 변수 상세 분석
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        analysis_parts.append("\n범주형 변수 상세 분석:")
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            analysis_parts.append(f"\n{col} 분포:")
            analysis_parts.append(f"  - 고유값: {len(value_counts)}개")
            analysis_parts.append(f"  - 최빈값: {value_counts.index[0]} ({value_counts.iloc[0]}개)")
            analysis_parts.append(f"  - 상위 5개 값:")
            for i, (value, count) in enumerate(value_counts.head(5).items(), 1):
                pct = (count / len(df)) * 100
                analysis_parts.append(f"    {i}. {value}: {count}개 ({pct:.1f}%)")
    else:
        analysis_parts.append("\n범주형 변수가 없습니다.")
    
    analysis_parts.append("")
    
    # 6. 상관관계 분석
    analysis_parts.append("=== 상관관계 분석 ===")
    if len(numeric_cols) >= 2:
        # 피어슨 상관계수
        pearson_corr = df[numeric_cols].corr()
        analysis_parts.append("피어슨 상관계수:")
        analysis_parts.append(pearson_corr.to_string())
        
        # 높은 상관관계 쌍 찾기
        high_corr_pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_val = pearson_corr.iloc[i, j]
                if abs(corr_val) > 0.7:  # 높은 상관관계 기준
                    high_corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_val))
        
        if high_corr_pairs:
            analysis_parts.append("\n높은 상관관계 (|r| > 0.7):")
            for var1, var2, corr_val in high_corr_pairs:
                analysis_parts.append(f"  {var1} ↔ {var2}: {corr_val:.3f}")
        else:
            analysis_parts.append("\n높은 상관관계를 보이는 변수 쌍이 없습니다.")
    else:
        analysis_parts.append("상관관계 분석을 위한 수치형 변수가 부족합니다 (최소 2개 필요).")
    
    analysis_parts.append("")
    
    # 7. 데이터 품질 평가
    analysis_parts.append("=== 데이터 품질 평가 ===")
    
    # 결측치 비율이 높은 컬럼
    high_missing_cols = []
    for col in df.columns:
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        if missing_pct > 50:
            high_missing_cols.append((col, missing_pct))
    
    if high_missing_cols:
        analysis_parts.append("⚠️  결측치가 많은 컬럼 (>50%):")
        for col, pct in high_missing_cols:
            analysis_parts.append(f"  - {col}: {pct:.1f}%")
    else:
        analysis_parts.append("✅ 결측치가 많은 컬럼이 없습니다.")
    
    # 이상치 가능성이 있는 수치형 변수
    outlier_candidates = []
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outlier_pct = (len(outliers) / len(df)) * 100
        
        if outlier_pct > 5:  # 이상치가 5% 이상인 경우
            outlier_candidates.append((col, outlier_pct))
    
    if outlier_candidates:
        analysis_parts.append("\n⚠️  이상치가 많은 수치형 변수 (>5%):")
        for col, pct in outlier_candidates:
            analysis_parts.append(f"  - {col}: {pct:.1f}%")
    else:
        analysis_parts.append("\n✅ 이상치가 많은 수치형 변수가 없습니다.")
    
    # 8. 전처리 권장사항
    analysis_parts.append("\n=== 전처리 권장사항 ===")
    
    # 결측치 처리 권장사항
    if total_missing > 0:
        analysis_parts.append("결측치 처리:")
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 50:
                analysis_parts.append(f"  - {col}: 결측치가 많으므로 삭제 고려")
            elif missing_pct > 0:
                if df[col].dtype in ['object', 'category']:
                    analysis_parts.append(f"  - {col}: 최빈값으로 대체 권장")
                else:
                    analysis_parts.append(f"  - {col}: 평균/중앙값으로 대체 권장")
    
    # 범주형 변수 인코딩 권장사항
    if len(categorical_cols) > 0:
        analysis_parts.append("\n범주형 변수 인코딩:")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count <= 10:
                analysis_parts.append(f"  - {col}: Label Encoding 권장 ({unique_count}개 고유값)")
            else:
                analysis_parts.append(f"  - {col}: One-Hot Encoding 고려 ({unique_count}개 고유값)")
    
    # 스케일링 권장사항
    if len(numeric_cols) > 0:
        analysis_parts.append("\n스케일링:")
        for col in numeric_cols:
            std_val = df[col].std()
            if std_val > 100:
                analysis_parts.append(f"  - {col}: 표준화 권장 (표준편차: {std_val:.1f})")
            else:
                analysis_parts.append(f"  - {col}: 정규화 고려 (표준편차: {std_val:.1f})")
    
    # 텍스트 분석 결과 생성
    text_analysis = "\n".join(analysis_parts)
    
    print("텍스트 분석 완료")
    print("text_analysis : ", text_analysis)
    
    return {
        "dataframe": df,
        "text_analysis": text_analysis
    }


# LangGraph 노드로 사용할 수 있는 함수
text_analysis_agent = RunnableLambda(analyze_data_text)
