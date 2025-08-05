"""
변수별 상관관계 분석 텍스트 에이전트
피어슨 상관계수, corr() 함수, 타겟 변수와의 상관관계 등을 분석하여 텍스트 형태로 결과를 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def analyze_correlations(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    변수 간 상관관계를 분석하고 텍스트 형태로 결과를 제공합니다.
    
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
            "correlation_analysis": "수치형 변수가 2개 미만이어서 상관관계 분석을 수행할 수 없습니다."
        }
    
    # 분석 결과를 텍스트로 변환
    analysis_text = "=== 변수별 상관관계 분석 ===\n\n"
    
    # 1. 전체 상관관계 행렬
    correlation_matrix = df[numeric_columns].corr()
    
    analysis_text += "📊 전체 상관관계 행렬:\n"
    analysis_text += correlation_matrix.to_string()
    analysis_text += "\n\n"
    
    # 2. 타겟 변수 식별 및 분석
    target_column = None
    for col in ['Survived', 'target', 'label', 'class', 'y']:
        if col in numeric_columns:
            target_column = col
            break
    
    if target_column:
        analysis_text += f"🎯 타겟 변수 ({target_column})와의 상관관계:\n"
        
        # 타겟 변수와의 상관관계
        target_correlations = correlation_matrix[target_column].sort_values(key=abs, ascending=False)
        
        for col in target_correlations.index:
            if col != target_column:
                corr_value = target_correlations[col]
                analysis_text += f"   - {col}: {corr_value:.4f}"
                
                # 상관관계 강도 해석
                if abs(corr_value) >= 0.7:
                    strength = "매우 강함"
                elif abs(corr_value) >= 0.5:
                    strength = "강함"
                elif abs(corr_value) >= 0.3:
                    strength = "보통"
                elif abs(corr_value) >= 0.1:
                    strength = "약함"
                else:
                    strength = "매우 약함"
                
                direction = "양의" if corr_value > 0 else "음의"
                analysis_text += f" ({direction} {strength})\n"
        
        analysis_text += "\n"
    
    # 3. 강한 상관관계 분석
    analysis_text += "🔍 강한 상관관계 분석 (|r| >= 0.5):\n"
    
    strong_correlations = []
    for i, col1 in enumerate(numeric_columns):
        for j, col2 in enumerate(numeric_columns[i+1:], i+1):
            corr_value = correlation_matrix.loc[col1, col2]
            if abs(corr_value) >= 0.5:
                strong_correlations.append({
                    'var1': col1,
                    'var2': col2,
                    'correlation': corr_value
                })
    
    if strong_correlations:
        # 절댓값 기준으로 정렬
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        for corr in strong_correlations:
            direction = "양의" if corr['correlation'] > 0 else "음의"
            analysis_text += f"   - {corr['var1']} ↔ {corr['var2']}: {corr['correlation']:.4f} ({direction} 강한 상관관계)\n"
    else:
        analysis_text += "   - 강한 상관관계(|r| >= 0.5)가 있는 변수 쌍이 없습니다.\n"
    
    analysis_text += "\n"
    
    # 4. 다중공선성 분석
    analysis_text += "⚠️ 다중공선성 분석:\n"
    
    # 상관계수 0.8 이상인 변수 쌍
    multicollinearity_pairs = []
    for i, col1 in enumerate(numeric_columns):
        for j, col2 in enumerate(numeric_columns[i+1:], i+1):
            corr_value = correlation_matrix.loc[col1, col2]
            if abs(corr_value) >= 0.8:
                multicollinearity_pairs.append({
                    'var1': col1,
                    'var2': col2,
                    'correlation': corr_value
                })
    
    if multicollinearity_pairs:
        analysis_text += "   - 다중공선성이 의심되는 변수 쌍 (|r| >= 0.8):\n"
        for pair in multicollinearity_pairs:
            analysis_text += f"     * {pair['var1']} ↔ {pair['var2']}: {pair['correlation']:.4f}\n"
        analysis_text += "   - 권장사항: 변수 선택 시 하나만 사용하거나, 주성분 분석 고려\n"
    else:
        analysis_text += "   - 심각한 다중공선성 문제는 발견되지 않았습니다.\n"
    
    analysis_text += "\n"
    
    # 5. 상관관계 패턴 분석
    analysis_text += "📈 상관관계 패턴 분석:\n"
    
    # 평균 상관관계
    mean_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
    analysis_text += f"   - 평균 상관관계: {mean_correlation:.4f}\n"
    
    # 상관관계 분포
    all_correlations = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
    positive_corr = (all_correlations > 0).sum()
    negative_corr = (all_correlations < 0).sum()
    zero_corr = (all_correlations == 0).sum()
    
    analysis_text += f"   - 양의 상관관계: {positive_corr}개\n"
    analysis_text += f"   - 음의 상관관계: {negative_corr}개\n"
    analysis_text += f"   - 무상관: {zero_corr}개\n"
    
    # 6. 특성 선택을 위한 상관관계 기반 권장사항
    analysis_text += "\n📋 특성 선택 권장사항:\n"
    
    if target_column:
        # 타겟 변수와의 상관관계가 높은 변수들
        high_target_corr = target_correlations[abs(target_correlations) >= 0.3].index.tolist()
        high_target_corr = [col for col in high_target_corr if col != target_column]
        
        if high_target_corr:
            analysis_text += f"   - 타겟 변수와 상관관계가 높은 변수들 (|r| >= 0.3):\n"
            for col in high_target_corr:
                corr_value = target_correlations[col]
                analysis_text += f"     * {col}: {corr_value:.4f}\n"
        else:
            analysis_text += "   - 타겟 변수와 상관관계가 높은 변수가 없습니다.\n"
    
    # 다중공선성이 있는 변수들
    if multicollinearity_pairs:
        analysis_text += "   - 다중공선성으로 인해 제거 고려할 변수들:\n"
        vars_to_remove = set()
        for pair in multicollinearity_pairs:
            if target_column and target_column in [pair['var1'], pair['var2']]:
                # 타겟 변수는 제거하지 않음
                other_var = pair['var2'] if pair['var1'] == target_column else pair['var1']
                vars_to_remove.add(other_var)
            else:
                # 상관계수가 더 낮은 변수 제거
                vars_to_remove.add(pair['var1'] if abs(pair['correlation']) < 0.9 else pair['var2'])
        
        for var in vars_to_remove:
            analysis_text += f"     * {var}\n"
    
    try:
        # LLM을 사용하여 추가 인사이트 생성
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=1000
        )
        
        llm_prompt = f"""
        다음은 데이터셋의 상관관계 분석 결과입니다:

        {analysis_text}

        위 분석 결과를 바탕으로 다음을 수행해주세요:

        1. **주요 발견사항**: 가장 중요한 상관관계 패턴들을 요약
        2. **비즈니스 인사이트**: 상관관계가 비즈니스적으로 의미하는 바
        3. **모델링 전략**: 상관관계를 고려한 모델링 전략 제안
        4. **특성 엔지니어링**: 상관관계를 활용한 특성 엔지니어링 아이디어

        다음 형식으로 답변해주세요:

        ## 주요 발견사항
        [가장 중요한 상관관계 패턴들]

        ## 비즈니스 인사이트
        [상관관계가 비즈니스적으로 의미하는 바]

        ## 모델링 전략
        [상관관계를 고려한 모델링 전략]

        ## 특성 엔지니어링 아이디어
        [상관관계를 활용한 특성 엔지니어링 제안]
        """
        
        response = llm.invoke([HumanMessage(content=llm_prompt)])
        llm_insights = response.content
        
        analysis_text += f"\n\n=== LLM 추가 인사이트 ===\n{llm_insights}"
        
    except Exception as e:
        print(f"LLM 분석 중 오류: {e}")
    
    print("상관관계 분석 완료")
    
    return {
        **inputs,
        "correlation_analysis": analysis_text
    }


# LangGraph 노드로 사용할 수 있는 함수
correlation_text_agent = RunnableLambda(analyze_correlations)
