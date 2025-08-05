"""
이상치 데이터 분석 텍스트 에이전트
사분위수 기반 방법, Z-Score 등을 사용하여 이상치를 분석하고 텍스트 형태로 결과를 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def analyze_outliers(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    이상치를 분석하고 텍스트 형태로 결과를 제공합니다.
    
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
            "outlier_analysis": "수치형 데이터가 없어서 이상치 분석을 수행할 수 없습니다."
        }
    
    # 분석 결과를 텍스트로 변환
    analysis_text = "=== 이상치 데이터 분석 ===\n\n"
    
    # 1. IQR 기반 이상치 분석
    analysis_text += "📊 IQR 기반 이상치 분석:\n"
    
    iqr_outliers = {}
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(df)) * 100
        
        iqr_outliers[col] = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': outlier_count,
            'outlier_percentage': outlier_percentage
        }
        
        analysis_text += f"   📈 {col}:\n"
        analysis_text += f"     - Q1: {Q1:.4f}, Q3: {Q3:.4f}, IQR: {IQR:.4f}\n"
        analysis_text += f"     - 하한: {lower_bound:.4f}, 상한: {upper_bound:.4f}\n"
        analysis_text += f"     - 이상치 개수: {outlier_count}개 ({outlier_percentage:.2f}%)\n"
        
        if outlier_count > 0:
            min_outlier = outliers[col].min()
            max_outlier = outliers[col].max()
            analysis_text += f"     - 이상치 범위: {min_outlier:.4f} ~ {max_outlier:.4f}\n"
        
        analysis_text += "\n"
    
    # 2. Z-Score 기반 이상치 분석
    analysis_text += "📊 Z-Score 기반 이상치 분석:\n"
    
    zscore_outliers = {}
    for col in numeric_columns:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers = df[z_scores > 3]  # Z-Score > 3을 이상치로 정의
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(df)) * 100
        
        zscore_outliers[col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'outlier_count': outlier_count,
            'outlier_percentage': outlier_percentage,
            'max_z_score': z_scores.max()
        }
        
        analysis_text += f"   📈 {col}:\n"
        analysis_text += f"     - 평균: {df[col].mean():.4f}, 표준편차: {df[col].std():.4f}\n"
        analysis_text += f"     - 이상치 개수 (|Z| > 3): {outlier_count}개 ({outlier_percentage:.2f}%)\n"
        analysis_text += f"     - 최대 Z-Score: {z_scores.max():.4f}\n"
        
        if outlier_count > 0:
            min_outlier = outliers[col].min()
            max_outlier = outliers[col].max()
            analysis_text += f"     - 이상치 범위: {min_outlier:.4f} ~ {max_outlier:.4f}\n"
        
        analysis_text += "\n"
    
    # 3. 이상치 패턴 분석
    analysis_text += "🔍 이상치 패턴 분석:\n"
    
    # IQR과 Z-Score 결과 비교
    for col in numeric_columns:
        iqr_count = iqr_outliers[col]['outlier_count']
        zscore_count = zscore_outliers[col]['outlier_count']
        
        analysis_text += f"   📈 {col}:\n"
        analysis_text += f"     - IQR 이상치: {iqr_count}개\n"
        analysis_text += f"     - Z-Score 이상치: {zscore_count}개\n"
        
        if iqr_count > zscore_count:
            analysis_text += f"     - IQR이 더 엄격한 기준 (더 많은 이상치 탐지)\n"
        elif zscore_count > iqr_count:
            analysis_text += f"     - Z-Score가 더 엄격한 기준 (더 많은 이상치 탐지)\n"
        else:
            analysis_text += f"     - 두 방법의 이상치 개수가 동일\n"
        
        # 이상치 비율에 따른 분류
        outlier_ratio = max(iqr_count, zscore_count) / len(df) * 100
        if outlier_ratio < 5:
            outlier_level = "낮음"
        elif outlier_ratio < 15:
            outlier_level = "보통"
        else:
            outlier_level = "높음"
        
        analysis_text += f"     - 이상치 수준: {outlier_level} ({outlier_ratio:.2f}%)\n\n"
    
    # 4. 이상치 처리 권장사항
    analysis_text += "📋 이상치 처리 권장사항:\n"
    
    for col in numeric_columns:
        iqr_count = iqr_outliers[col]['outlier_count']
        zscore_count = zscore_outliers[col]['outlier_count']
        outlier_ratio = max(iqr_count, zscore_count) / len(df) * 100
        
        analysis_text += f"   📈 {col}:\n"
        
        if outlier_ratio < 5:
            analysis_text += f"     - 이상치 비율이 낮음 ({outlier_ratio:.2f}%)\n"
            analysis_text += f"     - 권장사항: 제거 또는 중앙값으로 대체\n"
        elif outlier_ratio < 15:
            analysis_text += f"     - 이상치 비율이 보통 ({outlier_ratio:.2f}%)\n"
            analysis_text += f"     - 권장사항: 상한/하한값으로 대체 또는 로그 변환 고려\n"
        else:
            analysis_text += f"     - 이상치 비율이 높음 ({outlier_ratio:.2f}%)\n"
            analysis_text += f"     - 권장사항: 데이터 분포 재검토, 변수 변환 고려\n"
        
        # 분포 특성에 따른 추가 권장사항
        skewness = df[col].skew()
        if abs(skewness) > 1:
            analysis_text += f"     - 분포가 치우침 (왜도: {skewness:.2f})\n"
            analysis_text += f"     - 추가 권장사항: 로그 변환 또는 박스-콕스 변환 고려\n"
        
        analysis_text += "\n"
    
    # 5. 전체 이상치 요약
    analysis_text += "📊 전체 이상치 요약:\n"
    
    total_outliers_iqr = sum(iqr_outliers[col]['outlier_count'] for col in numeric_columns)
    total_outliers_zscore = sum(zscore_outliers[col]['outlier_count'] for col in numeric_columns)
    
    analysis_text += f"   - IQR 기반 총 이상치: {total_outliers_iqr}개\n"
    analysis_text += f"   - Z-Score 기반 총 이상치: {total_outliers_zscore}개\n"
    analysis_text += f"   - 수치형 변수 수: {len(numeric_columns)}개\n"
    
    # 이상치가 가장 많은 변수
    if total_outliers_iqr > 0:
        max_outlier_col = max(numeric_columns, key=lambda x: iqr_outliers[x]['outlier_count'])
        max_outlier_count = iqr_outliers[max_outlier_col]['outlier_count']
        analysis_text += f"   - 이상치가 가장 많은 변수: {max_outlier_col} ({max_outlier_count}개)\n"
    
    try:
        # LLM을 사용하여 추가 인사이트 생성
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=1000
        )
        
        llm_prompt = f"""
        다음은 데이터셋의 이상치 분석 결과입니다:

        {analysis_text}

        위 분석 결과를 바탕으로 다음을 수행해주세요:

        1. **주요 발견사항**: 가장 중요한 이상치 패턴들을 요약
        2. **이상치 원인 분석**: 이상치가 발생할 수 있는 비즈니스적 원인 추정
        3. **처리 전략**: 각 변수별 이상치 처리 전략 제안
        4. **모델링 영향**: 이상치가 모델링에 미칠 수 있는 영향 분석

        다음 형식으로 답변해주세요:

        ## 주요 발견사항
        [가장 중요한 이상치 패턴들]

        ## 이상치 원인 분석
        [이상치가 발생할 수 있는 비즈니스적 원인]

        ## 처리 전략
        [각 변수별 이상치 처리 전략]

        ## 모델링 영향
        [이상치가 모델링에 미칠 수 있는 영향]
        """
        
        response = llm.invoke([HumanMessage(content=llm_prompt)])
        llm_insights = response.content
        
        analysis_text += f"\n\n=== LLM 추가 인사이트 ===\n{llm_insights}"
        
    except Exception as e:
        print(f"LLM 분석 중 오류: {e}")
    
    print("이상치 분석 완료")
    
    return {
        **inputs,
        "outlier_analysis": analysis_text
    }


# LangGraph 노드로 사용할 수 있는 함수
outlier_text_agent = RunnableLambda(analyze_outliers)
