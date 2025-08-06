"""
RAG 기반 전처리 워크플로우 테스트
EDA 결과 → Planning Agent → KB RAG Agent → 코드 생성
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from planning_agents.planning_agent import planning_agent
from KB_rag_agents.KB_rag_agent import kb_rag_agent


def create_sample_eda_results() -> Dict[str, Any]:
    """샘플 EDA 결과 생성"""
    return {
        'text_analysis': """
        데이터셋 분석 결과:
        - 총 1000개 행, 15개 컬럼
        - 결측값이 여러 컬럼에 존재
        - 이상치가 수치형 변수에서 발견됨
        - 범주형 변수가 5개 있음
        - 높은 상관관계를 보이는 변수 쌍이 있음
        """,
        'null_analysis_text': """
        결측값 분석:
        - age 컬럼: 50개 결측값 (5%)
        - income 컬럼: 100개 결측값 (10%)
        - education 컬럼: 30개 결측값 (3%)
        - 결측값 패턴이 랜덤하지 않을 가능성 있음
        """,
        'outlier_analysis_text': """
        이상치 분석:
        - income 컬럼에서 극단적 이상치 발견
        - age 컬럼에서 일부 이상치 발견
        - 이상치 비율: income(8%), age(3%)
        """,
        'cate_analysis_text': """
        범주형 변수 분석:
        - gender: 2개 고유값
        - education: 5개 고유값
        - occupation: 12개 고유값
        - city: 25개 고유값
        - category: 8개 고유값
        """,
        'numeric_analysis_text': """
        수치형 변수 분석:
        - age: 평균 35.2, 표준편차 12.5
        - income: 평균 45000, 표준편차 25000
        - score: 평균 75.3, 표준편차 15.2
        - 스케일이 매우 다른 변수들이 존재
        """,
        'corr_analysis_text': """
        상관관계 분석:
        - age와 income: 0.65 (높은 상관관계)
        - income과 score: 0.45 (중간 상관관계)
        - age와 score: 0.32 (낮은 상관관계)
        - 다중공선성 문제 가능성 있음
        """
    }


def test_planning_agent():
    """Planning Agent 테스트"""
    print("=== Planning Agent 테스트 ===")
    
    # 샘플 EDA 결과 생성
    eda_results = create_sample_eda_results()
    
    # Planning Agent 실행
    inputs = {
        'text_analysis': eda_results['text_analysis'],
        'null_analysis_text': eda_results['null_analysis_text'],
        'outlier_analysis_text': eda_results['outlier_analysis_text'],
        'cate_analysis_text': eda_results['cate_analysis_text'],
        'numeric_analysis_text': eda_results['numeric_analysis_text'],
        'corr_analysis_text': eda_results['corr_analysis_text'],
    }
    
    result = planning_agent.invoke(inputs)
    
    print("Planning Agent 결과:")
    print(f"  작업 목록: {result['planning_info']['tasks']}")
    print(f"  우선순위: {result['planning_info']['priority']}")
    print(f"  근거: {result['planning_info']['rationale']}")
    print(f"  복잡도: {result['planning_info']['complexity']}")
    print(f"  예상 시간: {result['planning_info']['estimated_time']}")
    
    return result


def test_kb_rag_agent(planning_result: Dict[str, Any]):
    """KB RAG Agent 테스트"""
    print("\n=== KB RAG Agent 테스트 ===")
    
    # Planning 결과를 포함한 입력 생성
    inputs = {
        'text_analysis': create_sample_eda_results()['text_analysis'],
        'null_analysis_text': create_sample_eda_results()['null_analysis_text'],
        'outlier_analysis_text': create_sample_eda_results()['outlier_analysis_text'],
        'cate_analysis_text': create_sample_eda_results()['cate_analysis_text'],
        'numeric_analysis_text': create_sample_eda_results()['numeric_analysis_text'],
        'corr_analysis_text': create_sample_eda_results()['corr_analysis_text'],
        'planning_info': planning_result['planning_info']
    }
    
    # KB RAG Agent 실행
    result = kb_rag_agent.invoke(inputs)
    
    print("KB RAG Agent 결과:")
    print(f"  생성된 코드 길이: {len(result['generated_preprocessing_code'])} 문자")
    print(f"  관련 코드 수: {len(result['rag_relevant_codes'])}")
    
    # 관련 코드 정보 출력
    print("\n관련 코드 정보:")
    for i, code_info in enumerate(result['rag_relevant_codes'][:3]):
        print(f"  {i+1}. {code_info['category']} - {code_info['technique']['name']}")
        print(f"     유사도: {code_info['similarity']:.3f}")
        print(f"     설명: {code_info['technique']['description']}")
    
    return result


def test_complete_workflow():
    """전체 워크플로우 테스트"""
    print("=== RAG 기반 전처리 워크플로우 테스트 ===\n")
    
    # 1단계: Planning Agent
    planning_result = test_planning_agent()
    
    # 2단계: KB RAG Agent
    rag_result = test_kb_rag_agent(planning_result)
    
    # 3단계: 생성된 코드 출력
    print("\n=== 생성된 전처리 코드 ===")
    print(rag_result['generated_preprocessing_code'])
    
    return {
        'planning_result': planning_result,
        'rag_result': rag_result
    }


def test_with_sample_data():
    """샘플 데이터로 테스트"""
    print("\n=== 샘플 데이터로 테스트 ===")
    
    # 샘플 데이터 생성
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'age': np.random.normal(35, 12, n_samples),
        'income': np.random.normal(45000, 25000, n_samples),
        'score': np.random.normal(75, 15, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'city': np.random.choice(['Seoul', 'Busan', 'Incheon', 'Daegu'], n_samples)
    }
    
    # 결측값 추가
    data['age'][np.random.choice(n_samples, 5, replace=False)] = np.nan
    data['income'][np.random.choice(n_samples, 10, replace=False)] = np.nan
    
    # 이상치 추가
    data['income'][0] = 500000  # 극단적 이상치
    data['age'][1] = 150  # 이상치
    
    df = pd.DataFrame(data)
    
    print("샘플 데이터 정보:")
    print(f"  데이터 크기: {df.shape}")
    print(f"  결측값: {df.isnull().sum().sum()}개")
    print(f"  컬럼: {list(df.columns)}")
    print(f"  데이터 타입: {dict(df.dtypes)}")
    
    return df


if __name__ == "__main__":
    try:
        # 전체 워크플로우 테스트
        results = test_complete_workflow()
        
        # 샘플 데이터 테스트
        sample_df = test_with_sample_data()
        
        print("\n=== 테스트 완료 ===")
        print("모든 테스트가 성공적으로 완료되었습니다.")
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc() 