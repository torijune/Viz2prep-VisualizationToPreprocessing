"""
결측치 및 중복 데이터 분석 텍스트 에이전트
isnull, missing_data_percentage, duplicated 등을 분석하여 raw한 결측치 및 중복 데이터를 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda


def analyze_missing_and_duplicate_data(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    결측치와 중복 데이터를 분석하고 raw한 데이터를 제공합니다.
    
    Args:
        inputs: DataFrame이 포함된 입력 딕셔너리
        
    Returns:
        결측치 및 중복 데이터 분석 결과가 포함된 딕셔너리
    """
    df = inputs["dataframe"]
    
    # 결측치 분석
    total_missing = df.isnull().sum().sum()
    total_cells = df.size
    total_missing_percentage = (total_missing / total_cells) * 100
    
    # 컬럼별 결측치 분석
    missing_by_column = df.isnull().sum()
    missing_percentage_by_column = (missing_by_column / len(df)) * 100
    
    # 결측치가 있는 행들
    rows_with_missing = df.isnull().any(axis=1).sum()
    rows_with_missing_percentage = (rows_with_missing / len(df)) * 100
    
    # 결측치 패턴 (여러 컬럼에서 동시에 결측)
    missing_patterns = df.isnull().sum(axis=1).value_counts().sort_index()
    
    # 중복 데이터 분석
    total_duplicates = df.duplicated().sum()
    total_duplicates_percentage = (total_duplicates / len(df)) * 100
    
    # 중복 패턴 분석
    duplicate_info = {}
    if total_duplicates > 0:
        duplicate_counts = df.duplicated(keep=False).sum()
        duplicate_groups = df[df.duplicated(keep=False)].groupby(df.columns.tolist()).size()
        
        duplicate_info = {
            'duplicate_counts': int(duplicate_counts),
            'duplicate_groups_count': len(duplicate_groups),
            'most_common_duplicate_count': int(duplicate_groups.max()) if len(duplicate_groups) > 0 else 0
        }
    
    # 데이터 품질 통계
    quality_stats = {
        'total_missing': int(total_missing),
        'total_cells': int(total_cells),
        'total_missing_percentage': float(total_missing_percentage),
        'rows_with_missing': int(rows_with_missing),
        'rows_with_missing_percentage': float(rows_with_missing_percentage),
        'total_duplicates': int(total_duplicates),
        'total_duplicates_percentage': float(total_duplicates_percentage)
    }
    
    print("결측치 및 중복 데이터 분석 완료")
    
    # 분석 결과 로깅
    print("\n" + "="*50)
    print("📊 [EDA] 결측치 및 중복 데이터 분석 결과")
    print("="*50)
    print(f"📈 전체 데이터: {len(df)}행 x {len(df.columns)}열 = {total_cells}개 셀")
    print(f"🔍 결측치:")
    print(f"   - 전체 결측치: {total_missing}개 ({total_missing_percentage:.2f}%)")
    print(f"   - 결측치가 있는 행: {rows_with_missing}개 ({rows_with_missing_percentage:.2f}%)")
    
    # 결측치가 있는 컬럼들만 출력
    columns_with_missing = missing_by_column[missing_by_column > 0]
    if len(columns_with_missing) > 0:
        print(f"   - 결측치가 있는 컬럼: {len(columns_with_missing)}개")
        for col, count in columns_with_missing.items():
            percentage = missing_percentage_by_column[col]
            print(f"     * {col}: {count}개 ({percentage:.2f}%)")
    else:
        print(f"   - 결측치가 있는 컬럼: 없음")
    
    print(f"🔍 중복 데이터:")
    print(f"   - 전체 중복 행: {total_duplicates}개 ({total_duplicates_percentage:.2f}%)")
    if total_duplicates > 0:
        print(f"   - 중복 패턴이 있는 행: {duplicate_info['duplicate_counts']}개")
        print(f"   - 중복 그룹 수: {duplicate_info['duplicate_groups_count']}개")
        print(f"   - 가장 많이 중복된 패턴: {duplicate_info['most_common_duplicate_count']}번")
    
    result = {
        **inputs,
        "missing_duplicate_analysis": {
            "missing_by_column": missing_by_column.to_dict(),
            "missing_percentage_by_column": missing_percentage_by_column.to_dict(),
            "missing_patterns": missing_patterns.to_dict(),
            "duplicate_info": duplicate_info,
            "quality_stats": quality_stats,
            "columns_analyzed": df.columns.tolist()
        }
    }
    
    return result


# LangGraph 노드로 사용할 수 있는 함수
missing_duplicate_text_agent = RunnableLambda(analyze_missing_and_duplicate_data) 