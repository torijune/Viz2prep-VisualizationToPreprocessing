"""
분리된 에이전트들을 순차적으로 실행하는 테스트 파일
각 에이전트가 전문화된 역할을 수행하도록 구성
"""

import os
import pandas as pd
import numpy as np
from agents.data_loader import data_loader
from agents.statistics_agent import statistics_agent
from agents.visualization_agent import visualization_agent
from agents.business_context_agent import business_context_agent
from agents.feature_engineering_agent import feature_engineering_agent
from agents.preprocessing_strategy_agent import preprocessing_strategy_agent
from agents.preprocessing_agent import preprocessing_agent


def create_test_dataset():
    """
    테스트용 Titanic 데이터셋을 생성합니다.
    """
    np.random.seed(42)
    n_samples = 100
    
    # 기본 데이터 생성
    data = {
        'PassengerId': range(1, n_samples + 1),
        'Survived': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.3, 0.5]),
        'Name': [f'Passenger_{i}' for i in range(1, n_samples + 1)],
        'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.6, 0.4]),
        'Age': np.random.normal(30, 10, n_samples),
        'SibSp': np.random.poisson(0.5, n_samples),
        'Parch': np.random.poisson(0.3, n_samples),
        'Ticket': [f'Ticket_{i}' for i in range(1, n_samples + 1)],
        'Fare': np.random.exponential(30, n_samples),
        'Cabin': [f'Cabin_{i}' for i in range(1, n_samples + 1)],
        'Embarked': np.random.choice(['S', 'C', 'Q', 'n'], n_samples, p=[0.4, 0.2, 0.3, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # 결측값 추가
    df.loc[np.random.choice(df.index, 10), 'Age'] = np.nan
    df.loc[np.random.choice(df.index, 5), 'Fare'] = np.nan
    df.loc[np.random.choice(df.index, 65), 'Cabin'] = np.nan
    
    return df


def test_separated_agents():
    """
    분리된 에이전트들을 순차적으로 테스트합니다.
    """
    print("="*80)
    print("분리된 에이전트 순차 실행 테스트")
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
    
    # 4. 시각화 생성
    print("\n4. 시각화 생성 중...")
    viz_result = visualization_agent.invoke({"dataframe": df})
    plot_paths = viz_result['plot_paths']
    print(f"   생성된 플롯 수: {len(plot_paths)}")
    
    # 5. 비즈니스 컨텍스트 분석
    print("\n5. 비즈니스 컨텍스트 분석 중...")
    business_result = business_context_agent.invoke({
        "dataframe": df,
        "text_analysis": statistics_text
    })
    business_context = business_result['business_context']
    print(f"   비즈니스 컨텍스트 길이: {len(business_context)}")
    
    # 6. Feature Engineering 전략 생성
    print("\n6. Feature Engineering 전략 생성 중...")
    feature_result = feature_engineering_agent.invoke({
        "dataframe": df,
        "business_context": business_context,
        "text_analysis": statistics_text
    })
    feature_strategy = feature_result['feature_engineering_strategy']
    print(f"   Feature Engineering 전략 길이: {len(feature_strategy)}")
    
    # 7. 고급 전처리 전략 생성
    print("\n7. 고급 전처리 전략 생성 중...")
    preprocessing_strategy_result = preprocessing_strategy_agent.invoke({
        "dataframe": df,
        "business_context": business_context,
        "feature_engineering_strategy": feature_strategy,
        "text_analysis": statistics_text
    })
    preprocessing_strategy = preprocessing_strategy_result['preprocessing_strategy']
    print(f"   전처리 전략 길이: {len(preprocessing_strategy)}")
    
    # 8. 최종 전처리 실행
    print("\n8. 최종 전처리 실행 중...")
    print("   (OpenAI API 키가 필요합니다)")
    
    # OpenAI API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("   ⚠️  OpenAI API 키가 설정되지 않았습니다.")
        print("   env.example 파일을 .env로 복사하고 API 키를 설정해주세요.")
        return
    
    try:
        # 최종 전처리 에이전트 실행
        final_preprocessing_input = {
            "dataframe": df,
            "text_analysis": statistics_text,
            "business_context": business_context,
            "feature_engineering_strategy": feature_strategy,
            "preprocessing_strategy": preprocessing_strategy,
            "plot_paths": plot_paths
        }
        
        final_result = preprocessing_agent.invoke(final_preprocessing_input)
        
        # 결과 출력
        print("\n" + "="*80)
        print("분리된 에이전트 실행 결과")
        print("="*80)
        
        print(f"📊 원본 데이터 크기: {df.shape}")
        print(f"🔧 전처리된 데이터 크기: {final_result['preprocessed_dataframe'].shape}")
        
        if 'preprocessing_code' in final_result:
            code = final_result['preprocessing_code']
            print(f"💻 생성된 전처리 코드 길이: {len(code)} 문자")
        
        # 전처리 전후 비교
        print("\n📈 전처리 전후 비교:")
        print(f"   원본 결측값: {df.isnull().sum().sum()}")
        print(f"   전처리 후 결측값: {final_result['preprocessed_dataframe'].isnull().sum().sum()}")
        
        # 각 에이전트 결과 요약
        print("\n📋 각 에이전트 결과 요약:")
        print(f"   비즈니스 컨텍스트: {len(business_context)} 문자")
        print(f"   Feature Engineering 전략: {len(feature_strategy)} 문자")
        print(f"   전처리 전략: {len(preprocessing_strategy)} 문자")
        
        print("\n✅ 분리된 에이전트 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 최종 전처리 에이전트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 테스트 파일 정리
        if os.path.exists('test_titanic.csv'):
            os.remove('test_titanic.csv')


if __name__ == "__main__":
    test_separated_agents() 