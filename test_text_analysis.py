"""
텍스트 분석 에이전트 테스트
LLM이 데이터 구조와 특징을 빠르게 이해할 수 있도록 상세한 텍스트 요약을 생성하는 기능을 테스트합니다.
"""

import pandas as pd
import numpy as np
import os
from agents.data_loader import data_loader
from agents.text_analysis_agent import text_analysis_agent


def create_test_dataset():
    """
    테스트용 데이터셋을 생성합니다.
    """
    np.random.seed(42)
    
    # 다양한 특성을 가진 테스트 데이터 생성
    n_samples = 100
    
    # 수치형 변수들
    age = np.random.normal(30, 10, n_samples)
    age = np.clip(age, 1, 80)
    
    salary = np.random.exponential(50000, n_samples)
    salary = np.clip(salary, 20000, 200000)
    
    experience = np.random.poisson(5, n_samples)
    experience = np.clip(experience, 0, 20)
    
    # 범주형 변수들
    gender = np.random.choice(['Male', 'Female'], n_samples)
    department = np.random.choice(['IT', 'HR', 'Sales', 'Marketing', 'Finance'], n_samples)
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
    
    # 결측값 추가
    age[np.random.choice(n_samples, 5, replace=False)] = np.nan
    salary[np.random.choice(n_samples, 3, replace=False)] = np.nan
    department[np.random.choice(n_samples, 8, replace=False)] = np.nan
    education[np.random.choice(n_samples, 2, replace=False)] = np.nan
    
    # 이상치 추가
    age[np.random.choice(n_samples, 2, replace=False)] = [90, 95]  # 이상치
    salary[np.random.choice(n_samples, 1, replace=False)] = 500000  # 이상치
    
    # 타겟 변수 (성과 점수)
    performance = (
        (age < 40) * 0.3 +  # 젊은 사람은 성과 높음
        (salary > 80000) * 0.2 +  # 고급 임금은 성과 높음
        (experience > 5) * 0.3 +  # 경험 많은 사람은 성과 높음
        (education == 'Master') * 0.2  # 석사 학위는 성과 높음
    ) + np.random.normal(0, 0.1, n_samples)
    
    performance = np.clip(performance, 0, 1)
    
    # DataFrame 생성
    df = pd.DataFrame({
        'EmployeeID': range(1, n_samples + 1),
        'Age': age,
        'Gender': gender,
        'Department': department,
        'Education': education,
        'Experience': experience,
        'Salary': salary,
        'Performance': performance
    })
    
    return df


def test_text_analysis():
    """
    텍스트 분석 에이전트를 테스트합니다.
    """
    print("="*80)
    print("텍스트 분석 에이전트 테스트")
    print("="*80)
    
    # 1. 테스트 데이터셋 생성
    print("1. 테스트 데이터셋 생성 중...")
    test_df = create_test_dataset()
    test_df.to_csv('test_employee.csv', index=False)
    print(f"   생성된 데이터 크기: {test_df.shape}")
    print(f"   결측값 개수: {test_df.isnull().sum().sum()}")
    
    # 2. 데이터 로드
    print("\n2. 데이터 로드 중...")
    data_result = data_loader.invoke("test_employee.csv")
    df = data_result['dataframe']
    print(f"   로드된 데이터 크기: {df.shape}")
    
    # 3. 텍스트 분석 실행
    print("\n3. 텍스트 분석 실행 중...")
    text_result = text_analysis_agent.invoke({"dataframe": df})
    text_analysis = text_result['text_analysis']
    
    # 4. 결과 출력
    print("\n" + "="*80)
    print("텍스트 분석 결과")
    print("="*80)
    print(text_analysis)
    
    # 5. 요약 통계
    print("\n" + "="*80)
    print("분석 요약")
    print("="*80)
    print(f"📊 데이터 크기: {df.shape}")
    print(f"📝 텍스트 분석 길이: {len(text_analysis)} 문자")
    
    # 주요 발견사항 추출
    lines = text_analysis.split('\n')
    
    # 결측치 정보 추출
    missing_lines = [line for line in lines if '결측값' in line and '개' in line]
    if missing_lines:
        print(f"🔍 결측치가 있는 컬럼: {len(missing_lines)}개")
    
    # 범주형 변수 정보 추출
    categorical_lines = [line for line in lines if '범주형 변수' in line]
    if categorical_lines:
        print(f"📋 범주형 변수: {len(categorical_lines)}개")
    
    # 상관관계 정보 추출
    correlation_lines = [line for line in lines if '상관관계' in line]
    if correlation_lines:
        print(f"📈 상관관계 분석: 완료")
    
    # 전처리 권장사항 추출
    recommendation_lines = [line for line in lines if '권장' in line or '고려' in line]
    if recommendation_lines:
        print(f"💡 전처리 권장사항: {len(recommendation_lines)}개")
    
    print("\n✅ 텍스트 분석 테스트 완료!")
    
    # 테스트 파일 정리
    if os.path.exists('test_employee.csv'):
        os.remove('test_employee.csv')


if __name__ == "__main__":
    test_text_analysis() 