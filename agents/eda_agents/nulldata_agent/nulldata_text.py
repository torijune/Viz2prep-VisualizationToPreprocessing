"""
결측치 및 중복 데이터 분석 텍스트 에이전트
isnull, missing_data_percentage, duplicated 등을 분석하여 텍스트 형태로 결과를 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def analyze_missing_and_duplicate_data(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    결측치와 중복 데이터를 분석하고 텍스트 형태로 결과를 제공합니다.
    
    Args:
        inputs: DataFrame이 포함된 입력 딕셔너리
        
    Returns:
        결측치 및 중복 데이터 분석 결과가 포함된 딕셔너리
    """
    df = inputs["dataframe"]
    
    # 분석 결과를 텍스트로 변환
    analysis_text = "=== 결측치 및 중복 데이터 분석 ===\n\n"
    
    # 1. 결측치 분석
    analysis_text += "📊 결측치 분석:\n"
    
    # 전체 결측치 개수
    total_missing = df.isnull().sum().sum()
    total_cells = df.size
    total_missing_percentage = (total_missing / total_cells) * 100
    
    analysis_text += f"   - 전체 결측치 개수: {total_missing}개\n"
    analysis_text += f"   - 전체 데이터 셀 수: {total_cells}개\n"
    analysis_text += f"   - 전체 결측치 비율: {total_missing_percentage:.2f}%\n\n"
    
    # 컬럼별 결측치 분석
    missing_by_column = df.isnull().sum()
    missing_percentage_by_column = (missing_by_column / len(df)) * 100
    
    analysis_text += "   📋 컬럼별 결측치 분석:\n"
    for col in df.columns:
        missing_count = missing_by_column[col]
        missing_percentage = missing_percentage_by_column[col]
        
        if missing_count > 0:
            analysis_text += f"     - {col}: {missing_count}개 ({missing_percentage:.2f}%)\n"
        else:
            analysis_text += f"     - {col}: 결측치 없음\n"
    
    # 결측치 패턴 분석
    analysis_text += "\n   🔍 결측치 패턴 분석:\n"
    
    # 결측치가 있는 행들
    rows_with_missing = df.isnull().any(axis=1).sum()
    rows_with_missing_percentage = (rows_with_missing / len(df)) * 100
    
    analysis_text += f"     - 결측치가 있는 행: {rows_with_missing}개 ({rows_with_missing_percentage:.2f}%)\n"
    analysis_text += f"     - 완전한 행: {len(df) - rows_with_missing}개\n"
    
    # 결측치 패턴 (여러 컬럼에서 동시에 결측)
    missing_patterns = df.isnull().sum(axis=1).value_counts().sort_index()
    analysis_text += "     - 결측치 개수별 행 분포:\n"
    for missing_count, row_count in missing_patterns.items():
        if missing_count > 0:
            percentage = (row_count / len(df)) * 100
            analysis_text += f"       * {missing_count}개 결측: {row_count}행 ({percentage:.2f}%)\n"
    
    # 2. 중복 데이터 분석
    analysis_text += "\n📊 중복 데이터 분석:\n"
    
    # 전체 중복 행
    total_duplicates = df.duplicated().sum()
    total_duplicates_percentage = (total_duplicates / len(df)) * 100
    
    analysis_text += f"   - 전체 중복 행: {total_duplicates}개 ({total_duplicates_percentage:.2f}%)\n"
    
    # 중복 패턴 분석
    if total_duplicates > 0:
        duplicate_counts = df.duplicated(keep=False).sum()
        analysis_text += f"   - 중복 패턴이 있는 행: {duplicate_counts}개\n"
        
        # 중복 그룹 분석
        duplicate_groups = df[df.duplicated(keep=False)].groupby(df.columns.tolist()).size()
        analysis_text += f"   - 중복 그룹 수: {len(duplicate_groups)}개\n"
        
        # 가장 많이 중복된 패턴
        most_common_duplicate = duplicate_groups.idxmax()
        most_common_count = duplicate_groups.max()
        analysis_text += f"   - 가장 많이 중복된 패턴: {most_common_count}번\n"
    
    # 3. 데이터 품질 평가
    analysis_text += "\n📊 데이터 품질 평가:\n"
    
    # 결측치 기준으로 품질 평가
    if total_missing_percentage < 5:
        missing_quality = "우수"
    elif total_missing_percentage < 15:
        missing_quality = "양호"
    elif total_missing_percentage < 30:
        missing_quality = "보통"
    else:
        missing_quality = "불량"
    
    analysis_text += f"   - 결측치 품질: {missing_quality} ({total_missing_percentage:.2f}%)\n"
    
    # 중복 기준으로 품질 평가
    if total_duplicates_percentage < 1:
        duplicate_quality = "우수"
    elif total_duplicates_percentage < 5:
        duplicate_quality = "양호"
    elif total_duplicates_percentage < 15:
        duplicate_quality = "보통"
    else:
        duplicate_quality = "불량"
    
    analysis_text += f"   - 중복 데이터 품질: {duplicate_quality} ({total_duplicates_percentage:.2f}%)\n"
    
    # 4. 전처리 권장사항
    analysis_text += "\n📋 전처리 권장사항:\n"
    
    # 결측치가 많은 컬럼들
    high_missing_columns = missing_percentage_by_column[missing_percentage_by_column > 50].index.tolist()
    if high_missing_columns:
        analysis_text += f"   - 결측치가 50% 이상인 컬럼: {', '.join(high_missing_columns)}\n"
        analysis_text += "     → 제거 고려 또는 특별한 처리 필요\n"
    
    # 결측치가 적은 컬럼들
    low_missing_columns = missing_percentage_by_column[(missing_percentage_by_column > 0) & (missing_percentage_by_column <= 10)].index.tolist()
    if low_missing_columns:
        analysis_text += f"   - 결측치가 10% 이하인 컬럼: {', '.join(low_missing_columns)}\n"
        analysis_text += "     → 평균/중앙값/최빈값으로 대체 가능\n"
    
    # 중복 데이터 처리
    if total_duplicates > 0:
        analysis_text += f"   - 중복 데이터: {total_duplicates}개 제거 권장\n"
    
    try:
        # LLM을 사용하여 추가 인사이트 생성
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=1000
        )
        
        llm_prompt = f"""
        다음은 데이터셋의 결측치 및 중복 데이터 분석 결과입니다:

        {analysis_text}

        위 분석 결과를 바탕으로 다음을 수행해주세요:

        1. **주요 발견사항**: 가장 중요한 데이터 품질 이슈들을 요약
        2. **결측치 패턴 분석**: 결측치가 발생하는 패턴과 원인 추정
        3. **중복 데이터 분석**: 중복 데이터의 특성과 의미
        4. **전처리 전략**: 구체적인 결측치 및 중복 데이터 처리 방법 제안

        다음 형식으로 답변해주세요:

        ## 주요 발견사항
        [가장 중요한 데이터 품질 이슈들]

        ## 결측치 패턴 분석
        [결측치 발생 패턴과 원인 추정]

        ## 중복 데이터 분석
        [중복 데이터의 특성과 의미]

        ## 전처리 전략
        [구체적인 처리 방법 제안]
        """
        
        response = llm.invoke([HumanMessage(content=llm_prompt)])
        llm_insights = response.content
        
        analysis_text += f"\n\n=== LLM 추가 인사이트 ===\n{llm_insights}"
        
    except Exception as e:
        print(f"LLM 분석 중 오류: {e}")
    
    print("결측치 및 중복 데이터 분석 완료")
    
    return {
        **inputs,
        "missing_duplicate_analysis": analysis_text
    }


# LangGraph 노드로 사용할 수 있는 함수
missing_duplicate_text_agent = RunnableLambda(analyze_missing_and_duplicate_data) 