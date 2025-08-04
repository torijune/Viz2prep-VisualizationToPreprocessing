"""
도메인 분석 에이전트
LLM이 데이터의 비즈니스 컨텍스트와 도메인 지식을 활용하여 feature engineering 인사이트를 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def analyze_domain_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM을 사용하여 데이터의 도메인 컨텍스트를 분석하고 feature engineering 인사이트를 제공합니다.
    
    Args:
        inputs: DataFrame과 텍스트 분석이 포함된 입력 딕셔너리
        
    Returns:
        도메인 분석 결과가 포함된 딕셔너리
    """
    df = inputs["dataframe"]
    text_analysis = inputs.get("text_analysis", "")
    
    # 타겟 변수와 주요 변수들 식별
    target_column = None
    for col in ['Survived', 'target', 'label', 'class', 'y']:
        if col in df.columns:
            target_column = col
            break
    
    if not target_column:
        target_column = df.columns[-1]  # 마지막 컬럼을 타겟으로 사용
    
    # 주요 변수들 추출
    key_variables = []
    for col in df.columns:
        if col != target_column:
            if df[col].dtype in ['object', 'category']:
                key_variables.append(f"{col} (범주형)")
            else:
                key_variables.append(f"{col} (수치형)")
    
    # 도메인 분석을 위한 프롬프트 생성
    domain_prompt = f"""
    당신은 데이터 과학자이자 도메인 전문가입니다. 
    다음 데이터셋의 비즈니스 컨텍스트를 분석하고, 도메인 지식을 바탕으로 고급 feature engineering 전략을 제안해주세요.

    === 데이터셋 정보 ===
    - 타겟 변수: {target_column}
    - 주요 변수들: {', '.join(key_variables)}
    - 데이터 크기: {df.shape[0]} 행 x {df.shape[1]} 열

    === 텍스트 분석 결과 ===
    {text_analysis}

    === 도메인 분석 요청사항 ===

    1. **비즈니스 컨텍스트 분석**
    - 이 데이터셋이 어떤 비즈니스 문제를 해결하려는지 분석
    - 각 변수의 비즈니스적 의미와 중요도 평가

    2. **도메인 지식 기반 Feature Engineering 제안**
    - 기존 변수들을 조합한 새로운 특성 생성 방안
    - 비즈니스 로직에 기반한 파생 변수 제안
    - 도메인 전문가가 고려할 수 있는 숨겨진 패턴이나 관계

    3. **고급 전처리 전략**
    - 이상치 처리: 비즈니스 관점에서의 이상치 정의와 처리 방법
    - 결측치 처리: 도메인 지식을 활용한 결측치 대체 전략
    - 스케일링: 비즈니스 컨텍스트에 적합한 정규화 방법

    4. **특성 선택 및 우선순위**
    - 비즈니스 중요도에 따른 특성 우선순위
    - 제거해야 할 특성과 보존해야 할 특성 구분

    5. **모델링 전략 제안**
    - 도메인 특성에 적합한 모델 선택 가이드
    - 비즈니스 요구사항을 반영한 평가 지표 제안

    다음 형식으로 답변해주세요:

    ## 비즈니스 컨텍스트 분석
    [비즈니스 문제와 데이터의 의미 분석]

    ## 도메인 기반 Feature Engineering
    ### 1. 파생 변수 생성
    - [구체적인 파생 변수와 생성 방법]

    ### 2. 특성 조합
    - [기존 변수들의 조합 방법]

    ### 3. 비즈니스 로직 기반 전처리
    - [도메인 지식을 활용한 전처리 방법]

    ## 고급 전처리 전략
    ### 1. 이상치 처리
    - [비즈니스 관점의 이상치 정의와 처리]

    ### 2. 결측치 처리
    - [도메인 지식 기반 결측치 대체]

    ### 3. 스케일링 및 정규화
    - [비즈니스 컨텍스트에 적합한 방법]

    ## 특성 선택 및 우선순위
    - [중요도별 특성 분류]

    ## 모델링 전략
    - [도메인 특성에 적합한 모델과 평가 방법]
    """

    try:
        # OpenAI LLM 호출
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=2000
        )
        
        response = llm.invoke([HumanMessage(content=domain_prompt)])
        domain_analysis = response.content
        
        print("도메인 분석 완료")
        print("domain_analysis : ", domain_analysis)
        
        return {
            **inputs,
            "domain_analysis": domain_analysis
        }
        
    except Exception as e:
        print(f"도메인 분석 오류: {e}")
        # 기본 도메인 분석 제공
        basic_analysis = f"""
        ## 기본 도메인 분석
        타겟 변수: {target_column}
        주요 변수: {', '.join(key_variables)}

        ### 기본 Feature Engineering 제안:
        1. 범주형 변수 인코딩
        2. 수치형 변수 스케일링
        3. 결측치 처리
        4. 이상치 탐지 및 처리
        """
                
        return {
            **inputs,
            "domain_analysis": basic_analysis
        }


# LangGraph 노드로 사용할 수 있는 함수
domain_analysis_agent = RunnableLambda(analyze_domain_context)
