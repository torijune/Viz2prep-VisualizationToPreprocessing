"""
멀티모달 전처리 에이전트 테스트
통계 정보와 시각화를 바탕으로 전처리 코드를 생성하는 기능을 테스트합니다.
"""

import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from agents.data_loader import data_loader
from agents.statistics_agent import statistics_agent
from agents.visualization_agent import visualization_agent
from agents.preprocessing_agent import preprocessing_agent

# 환경 변수 로드
load_dotenv()


def create_test_dataset():
    """
    테스트용 데이터셋을 생성합니다.
    """
    np.random.seed(42)
    
    # Titanic 데이터와 유사한 구조의 테스트 데이터 생성
    n_samples = 100
    
    # 수치형 변수들
    age = np.random.normal(30, 10, n_samples)
    age = np.clip(age, 1, 80)  # 나이 범위 제한
    
    fare = np.random.exponential(30, n_samples)
    fare = np.clip(fare, 0, 500)  # 요금 범위 제한
    
    # 범주형 변수들
    sex = np.random.choice(['male', 'female'], n_samples)
    pclass = np.random.choice([1, 2, 3], n_samples)
    embarked = np.random.choice(['S', 'C', 'Q'], n_samples)
    
    # 결측값 추가
    age[np.random.choice(n_samples, 10, replace=False)] = np.nan
    fare[np.random.choice(n_samples, 5, replace=False)] = np.nan
    embarked[np.random.choice(n_samples, 8, replace=False)] = np.nan
    
    # 타겟 변수 (생존 여부)
    # 나이, 성별, 요금, 클래스에 기반한 간단한 로직
    survival_prob = (
        (age < 30) * 0.8 +  # 젊은 사람은 생존 확률 높음
        (sex == 'female') * 0.7 +  # 여성은 생존 확률 높음
        (pclass == 1) * 0.6 +  # 1등급은 생존 확률 높음
        (fare > 50) * 0.4  # 고급 요금은 생존 확률 높음
    ) / 4
    
    survived = np.random.binomial(1, np.clip(survival_prob, 0, 1))
    
    # DataFrame 생성
    df = pd.DataFrame({
        'PassengerId': range(1, n_samples + 1),
        'Survived': survived,
        'Pclass': pclass,
        'Name': [f'Passenger_{i}' for i in range(1, n_samples + 1)],
        'Sex': sex,
        'Age': age,
        'SibSp': np.random.poisson(1, n_samples),
        'Parch': np.random.poisson(0.5, n_samples),
        'Ticket': [f'Ticket_{i}' for i in range(1, n_samples + 1)],
        'Fare': fare,
        'Cabin': [f'Cabin_{i}' if np.random.random() > 0.7 else np.nan for i in range(1, n_samples + 1)],
        'Embarked': embarked
    })
    
    return df


def test_multimodal_preprocessing():
    """
    멀티모달 전처리 에이전트를 테스트합니다.
    """
    print("="*80)
    print("멀티모달 전처리 에이전트 테스트")
    print("="*80)
    
    # 1. 테스트 데이터셋 생성
    print("1. 테스트 데이터셋 생성 중...")
    test_df = create_test_dataset()
    test_df.to_csv('test_titanic.csv', index=False)
    print(f"   생성된 데이터 크기: {test_df.shape}")
    print(f"   결측값 개수: {test_df.isnull().sum().sum()}")
    
    # 2. 데이터 로드
    print("\n2. 데이터 로드 중...")
    data_result = data_loader.invoke({"file_path": "test_titanic.csv"})
    df = data_result['dataframe']
    print(f"   로드된 데이터 크기: {df.shape}")
    
    # 3. 통계 정보 생성
    print("\n3. 통계 정보 생성 중...")
    stats_result = statistics_agent.invoke({"dataframe": df})
    statistics_text = stats_result['statistics_text']
    print(f"   통계 텍스트 길이: {len(statistics_text)}")
    print("   통계 정보 미리보기:")
    print("   " + statistics_text[:200] + "...")
    
    # 4. 시각화 생성
    print("\n4. 시각화 생성 중...")
    viz_result = visualization_agent.invoke({"dataframe": df})
    plot_paths = viz_result['plot_paths']
    print(f"   생성된 플롯 수: {len(plot_paths)}")
    for i, path in enumerate(plot_paths, 1):
        if os.path.exists(path):
            print(f"   {i}. {path} ({os.path.getsize(path)} bytes)")
    
    # 5. 멀티모달 전처리 에이전트 실행
    print("\n5. 멀티모달 전처리 에이전트 실행 중...")
    print("   (OpenAI API 키가 필요합니다)")
    
    # OpenAI API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("   ⚠️  OpenAI API 키가 설정되지 않았습니다.")
        print("   env.example 파일을 .env로 복사하고 API 키를 설정해주세요.")
        return
    
    try:
        # 전처리 에이전트 실행
        preprocessing_input = {
            "dataframe": df,
            "text_analysis": statistics_text,
            "plot_paths": plot_paths
        }
        
        preprocessing_result = preprocessing_agent.invoke(preprocessing_input)
        
        # 결과 출력
        print("\n" + "="*80)
        print("멀티모달 전처리 결과")
        print("="*80)
        
        print(f"📊 원본 데이터 크기: {df.shape}")
        print(f"🔧 전처리된 데이터 크기: {preprocessing_result['preprocessed_dataframe'].shape}")
        
        if 'preprocessing_code' in preprocessing_result:
            code = preprocessing_result['preprocessing_code']
            print(f"💻 생성된 전처리 코드 길이: {len(code)} 문자")
            print("\n생성된 전처리 코드:")
            print("-" * 50)
            print(code)
            print("-" * 50)
        
        # 전처리 전후 비교
        print("\n📈 전처리 전후 비교:")
        print(f"   원본 결측값: {df.isnull().sum().sum()}")
        print(f"   전처리 후 결측값: {preprocessing_result['preprocessed_dataframe'].isnull().sum().sum()}")
        
        print("\n✅ 멀티모달 전처리 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 전처리 에이전트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 테스트 파일 정리
        if os.path.exists('test_titanic.csv'):
            os.remove('test_titanic.csv')


if __name__ == "__main__":
    test_multimodal_preprocessing() 