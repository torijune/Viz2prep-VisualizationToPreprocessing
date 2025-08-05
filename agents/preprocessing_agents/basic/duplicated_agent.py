"""
중복 데이터 처리 전처리 에이전트
데이터의 중복 행을 탐지하고 처리합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableLambda


def handle_duplicates(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    중복 데이터를 탐지하고 처리하는 전처리 함수
    
    Args:
        inputs: DataFrame과 처리 방법이 포함된 입력 딕셔너리
        
    Returns:
        중복이 처리된 DataFrame이 포함된 딕셔너리
    """
    df = inputs["dataframe"].copy()
    method = inputs.get("duplicate_method", "auto")  # auto, remove, keep_first, keep_last
    subset = inputs.get("duplicate_subset", None)  # 특정 컬럼만 고려
    keep = inputs.get("duplicate_keep", "first")  # first, last, False
    
    print(f"중복 데이터 처리 시작: {method} 방법")
    
    # 중복 현황 분석
    original_count = len(df)
    duplicate_count = df.duplicated(subset=subset).sum()
    duplicate_percentage = (duplicate_count / original_count) * 100
    
    print(f"  원본 데이터: {original_count}행")
    print(f"  중복 행: {duplicate_count}행 ({duplicate_percentage:.1f}%)")
    
    if subset:
        print(f"  중복 판단 기준: {subset}")
    
    duplicate_info = {
        'original_count': original_count,
        'duplicate_count': duplicate_count,
        'duplicate_percentage': duplicate_percentage,
        'subset': subset
    }
    
    if duplicate_count == 0:
        print("  중복 데이터가 없습니다.")
        return {
            **inputs,
            "dataframe": df,
            "duplicate_info": duplicate_info
        }
    
    if method == "auto":
        # 자동 처리: 중복 비율에 따라 결정
        if duplicate_percentage > 50:
            # 중복이 50% 이상인 경우 경고
            print("  ⚠️  중복 데이터가 많습니다. 수동 검토를 권장합니다.")
            return {
                **inputs,
                "dataframe": df,
                "duplicate_info": duplicate_info,
                "warning": "중복 데이터가 50% 이상입니다. 수동 검토를 권장합니다."
            }
        else:
            # 중복 제거 (첫 번째 유지)
            df = df.drop_duplicates(subset=subset, keep='first')
            print(f"  → 중복 행 {duplicate_count}개 제거 (첫 번째 유지)")
    
    elif method == "remove":
        # 모든 중복 제거
        df = df.drop_duplicates(subset=subset, keep=False)
        print(f"  → 중복 행 {duplicate_count}개 제거 (모든 중복 제거)")
    
    elif method == "keep_first":
        # 첫 번째 유지
        df = df.drop_duplicates(subset=subset, keep='first')
        print(f"  → 중복 행 {duplicate_count}개 제거 (첫 번째 유지)")
    
    elif method == "keep_last":
        # 마지막 유지
        df = df.drop_duplicates(subset=subset, keep='last')
        print(f"  → 중복 행 {duplicate_count}개 제거 (마지막 유지)")
    
    # 처리 후 결과
    final_count = len(df)
    removed_count = original_count - final_count
    
    duplicate_info.update({
        'final_count': final_count,
        'removed_count': removed_count,
        'removal_percentage': (removed_count / original_count) * 100
    })
    
    print(f"중복 처리 완료: 최종 {final_count}행 (제거: {removed_count}행)")
    
    return {
        **inputs,
        "dataframe": df,
        "duplicate_info": duplicate_info
    }


# LangGraph 노드로 사용할 수 있는 함수
duplicated_agent = RunnableLambda(handle_duplicates)