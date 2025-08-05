"""
범주형 데이터 분석 텍스트 에이전트
unique 값, value_counts, 카테고리 불균형 등을 분석하여 텍스트 형태로 결과를 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def analyze_categorical_data(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    범주형 데이터를 분석하고 텍스트 형태로 결과를 제공합니다.
    
    Args:
        inputs: DataFrame이 포함된 입력 딕셔너리
        
    Returns:
        범주형 데이터 분석 결과가 포함된 딕셔너리
    """
    df = inputs["dataframe"]
    
    # 범주형 컬럼 선택 (object, category 타입)
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_columns:
        return {
            **inputs,
            "categorical_analysis": "범주형 데이터가 없습니다."
        }
    
    # 분석 결과를 텍스트로 변환
    analysis_text = "=== 범주형 데이터 분석 ===\n\n"
    
    for col in categorical_columns:
        analysis_text += f"📊 {col} 컬럼 분석:\n"
        
        # 기본 정보
        unique_count = df[col].nunique()
        total_count = len(df[col])
        missing_count = df[col].isnull().sum()
        missing_percentage = (missing_count / total_count) * 100
        
        analysis_text += f"   - 고유값 개수: {unique_count}개\n"
        analysis_text += f"   - 총 데이터 개수: {total_count}개\n"
        analysis_text += f"   - 결측값: {missing_count}개 ({missing_percentage:.1f}%)\n"
        
        # value_counts 분석
        value_counts = df[col].value_counts()
        analysis_text += f"   - 상위 5개 값:\n"
        for i, (value, count) in enumerate(value_counts.head().items()):
            percentage = (count / total_count) * 100
            analysis_text += f"     {i+1}. {value}: {count}개 ({percentage:.1f}%)\n"
        
        # 카테고리 불균형 분석
        if unique_count > 1:
            # 엔트로피 계산 (불균형 지표)
            proportions = value_counts / total_count
            entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
            max_entropy = np.log2(unique_count)
            balance_ratio = entropy / max_entropy if max_entropy > 0 else 0
            
            analysis_text += f"   - 카테고리 불균형 분석:\n"
            analysis_text += f"     * 엔트로피: {entropy:.3f}\n"
            analysis_text += f"     * 최대 엔트로피: {max_entropy:.3f}\n"
            analysis_text += f"     * 균형 비율: {balance_ratio:.3f}\n"
            
            if balance_ratio > 0.8:
                balance_status = "균형잡힌 분포"
            elif balance_ratio > 0.5:
                balance_status = "보통 분포"
            else:
                balance_status = "불균형 분포"
            
            analysis_text += f"     * 분포 상태: {balance_status}\n"
        
        # 카디널리티 분석
        if unique_count <= 10:
            cardinality_status = "낮음 (10개 이하)"
        elif unique_count <= 50:
            cardinality_status = "보통 (11-50개)"
        else:
            cardinality_status = "높음 (50개 초과)"
        
        analysis_text += f"   - 카디널리티: {cardinality_status}\n"
        
        # 결측값 패턴 분석
        if missing_count > 0:
            analysis_text += f"   - 결측값 패턴: {missing_count}개 결측\n"
        
        analysis_text += "\n"
    
    # 전체 범주형 데이터 요약
    analysis_text += "=== 전체 범주형 데이터 요약 ===\n"
    analysis_text += f"- 범주형 컬럼 수: {len(categorical_columns)}개\n"
    analysis_text += f"- 컬럼 목록: {', '.join(categorical_columns)}\n"
    
    # 범주형 변수 간 관계 분석 (범주형 컬럼이 2개 이상인 경우)
    if len(categorical_columns) >= 2:
        analysis_text += f"\n=== 범주형 변수 간 관계 분석 ===\n"
        
        # 카이제곱 검정을 위한 교차표 생성
        for i, col1 in enumerate(categorical_columns):
            for j, col2 in enumerate(categorical_columns[i+1:], i+1):
                cross_table = pd.crosstab(df[col1], df[col2], margins=True)
                analysis_text += f"\n{col1} vs {col2} 교차표:\n"
                analysis_text += cross_table.to_string()
                analysis_text += "\n"
    
    try:
        # LLM을 사용하여 추가 인사이트 생성
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=1000
        )
        
        llm_prompt = f"""
        다음은 데이터셋의 범주형 변수 분석 결과입니다:

        {analysis_text}

        위 분석 결과를 바탕으로 다음을 수행해주세요:

        1. **주요 발견사항**: 가장 중요한 범주형 데이터 특징들을 요약
        2. **불균형 분석**: 카테고리 불균형이 심한 변수들과 그 의미
        3. **인코딩 전략**: 각 범주형 변수에 적합한 인코딩 방법 제안
        4. **전처리 권장사항**: 범주형 변수 전처리를 위한 구체적인 방법 제안

        다음 형식으로 답변해주세요:

        ## 주요 발견사항
        [가장 중요한 범주형 데이터 특징들]

        ## 불균형 분석
        [카테고리 불균형이 심한 변수들과 의미]

        ## 인코딩 전략
        [각 범주형 변수별 적합한 인코딩 방법]

        ## 전처리 권장사항
        [범주형 변수 전처리를 위한 구체적인 방법]
        """
        
        response = llm.invoke([HumanMessage(content=llm_prompt)])
        llm_insights = response.content
        
        analysis_text += f"\n\n=== LLM 추가 인사이트 ===\n{llm_insights}"
        
    except Exception as e:
        print(f"LLM 분석 중 오류: {e}")
    
    print("범주형 데이터 분석 완료")
    
    return {
        **inputs,
        "categorical_analysis": analysis_text
    }


# LangGraph 노드로 사용할 수 있는 함수
categorical_text_agent = RunnableLambda(analyze_categorical_data)
