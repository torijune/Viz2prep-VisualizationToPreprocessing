"""
연속형 데이터 통계치 분석 텍스트 에이전트
Min, Max, Mean, 분포 왜도, 첨도 등을 분석하여 텍스트 형태로 결과를 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def analyze_numeric_statistics(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    연속형 데이터의 통계치를 분석하고 텍스트 형태로 결과를 제공합니다.
    
    Args:
        inputs: DataFrame이 포함된 입력 딕셔너리
        
    Returns:
        연속형 데이터 통계 분석 결과가 포함된 딕셔너리
    """
    df = inputs["dataframe"]
    
    # 수치형 컬럼만 선택
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_columns:
        return {
            **inputs,
            "numeric_analysis": "수치형 데이터가 없습니다."
        }
    
    # 기본 통계 분석
    stats_analysis = {}
    for col in numeric_columns:
        stats = df[col].describe()
        skewness = df[col].skew()
        kurtosis = df[col].kurtosis()
        
        stats_analysis[col] = {
            'count': stats['count'],
            'mean': stats['mean'],
            'std': stats['std'],
            'min': stats['min'],
            '25%': stats['25%'],
            '50%': stats['50%'],
            '75%': stats['75%'],
            'max': stats['max'],
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    # 분석 결과를 텍스트로 변환
    analysis_text = "=== 연속형 데이터 통계 분석 ===\n\n"
    
    for col, stats in stats_analysis.items():
        analysis_text += f"📊 {col} 컬럼 분석:\n"
        analysis_text += f"   - 기본 통계: count={stats['count']:.0f}, mean={stats['mean']:.2f}, std={stats['std']:.2f}\n"
        analysis_text += f"   - 범위: min={stats['min']:.2f}, max={stats['max']:.2f}\n"
        analysis_text += f"   - 사분위수: Q1={stats['25%']:.2f}, Q2={stats['50%']:.2f}, Q3={stats['75%']:.2f}\n"
        analysis_text += f"   - 분포 특성: 왜도={stats['skewness']:.2f}, 첨도={stats['kurtosis']:.2f}\n"
        
        # 분포 해석
        if abs(stats['skewness']) < 0.5:
            skew_interpretation = "대칭 분포"
        elif stats['skewness'] > 0:
            skew_interpretation = "오른쪽으로 치우친 분포"
        else:
            skew_interpretation = "왼쪽으로 치우친 분포"
        
        if stats['kurtosis'] > 3:
            kurt_interpretation = "뾰족한 분포 (첨도 높음)"
        elif stats['kurtosis'] < 3:
            kurt_interpretation = "넓은 분포 (첨도 낮음)"
        else:
            kurt_interpretation = "정상 분포"
        
        analysis_text += f"   - 분포 해석: {skew_interpretation}, {kurt_interpretation}\n\n"
    
    # 전체 수치형 데이터 요약
    analysis_text += "=== 전체 수치형 데이터 요약 ===\n"
    analysis_text += f"- 수치형 컬럼 수: {len(numeric_columns)}개\n"
    analysis_text += f"- 컬럼 목록: {', '.join(numeric_columns)}\n"
    
    # 상관관계 분석 (수치형 컬럼이 2개 이상인 경우)
    if len(numeric_columns) >= 2:
        correlation_matrix = df[numeric_columns].corr()
        analysis_text += f"\n=== 수치형 변수 간 상관관계 ===\n"
        analysis_text += correlation_matrix.to_string()
    
    try:
        # LLM을 사용하여 추가 인사이트 생성
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=1000
        )
        
        llm_prompt = f"""
        다음은 데이터셋의 수치형 변수 통계 분석 결과입니다:

        {analysis_text}

        위 분석 결과를 바탕으로 다음을 수행해주세요:

        1. **주요 발견사항**: 가장 중요한 통계적 특징들을 요약
        2. **분포 특성**: 각 변수의 분포 형태와 의미 해석
        3. **이상치 가능성**: 통계치를 바탕으로 이상치가 있을 수 있는 변수 식별
        4. **전처리 권장사항**: 스케일링, 정규화 등의 전처리 방법 제안

        다음 형식으로 답변해주세요:

        ## 주요 발견사항
        [가장 중요한 통계적 특징들]

        ## 분포 특성 분석
        [각 변수의 분포 형태와 의미]

        ## 이상치 탐지 힌트
        [이상치가 있을 가능성이 높은 변수들]

        ## 전처리 권장사항
        [수치형 변수에 대한 전처리 방법 제안]
        """
        
        response = llm.invoke([HumanMessage(content=llm_prompt)])
        llm_insights = response.content
        
        analysis_text += f"\n\n=== LLM 추가 인사이트 ===\n{llm_insights}"
        
    except Exception as e:
        print(f"LLM 분석 중 오류: {e}")
    
    print("연속형 데이터 통계 분석 완료")
    
    return {
        **inputs,
        "numeric_analysis": analysis_text
    }


# LangGraph 노드로 사용할 수 있는 함수
numeric_text_agent = RunnableLambda(analyze_numeric_statistics)
