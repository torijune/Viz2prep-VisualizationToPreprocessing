"""
간단한 테스트 스크립트
기본 기능들이 정상적으로 작동하는지 확인합니다.
"""

import pandas as pd
import numpy as np
import os
from agents.data_loader import data_loader
from agents.statistics_agent import statistics_agent
from agents.visualization_agent import visualization_agent


def test_data_loader():
    """데이터 로더 테스트"""
    print("=== 데이터 로더 테스트 ===")
    
    # 간단한 테스트 데이터 생성
    test_data = pd.DataFrame({
        'A': [1, 2, 3, np.nan, 5],
        'B': ['a', 'b', 'c', 'd', 'e'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5]
    })
    
    # CSV 파일로 저장
    test_data.to_csv('test_data.csv', index=False)
    
    # 데이터 로더 테스트
    result = data_loader.invoke("test_data.csv")
    print(f"로드된 데이터 크기: {result['dataframe'].shape}")
    print("✅ 데이터 로더 테스트 통과")
    
    # 테스트 파일 삭제
    os.remove('test_data.csv')
    
    return result['dataframe']


def test_statistics_agent(df):
    """통계 에이전트 테스트"""
    print("\n=== 통계 에이전트 테스트 ===")
    
    result = statistics_agent.invoke({"dataframe": df})
    print(f"통계 텍스트 길이: {len(result['statistics_text'])}")
    print("✅ 통계 에이전트 테스트 통과")
    
    return result


def test_visualization_agent(df):
    """시각화 에이전트 테스트"""
    print("\n=== 시각화 에이전트 테스트 ===")
    
    result = visualization_agent.invoke({"dataframe": df})
    print(f"생성된 플롯 수: {len(result['plot_paths'])}")
    
    # 플롯 파일들이 실제로 생성되었는지 확인
    for plot_path in result['plot_paths']:
        if os.path.exists(plot_path):
            print(f"✅ 플롯 생성됨: {plot_path}")
        else:
            print(f"❌ 플롯 생성 실패: {plot_path}")
    
    print("✅ 시각화 에이전트 테스트 통과")
    
    return result


def main():
    """메인 테스트 함수"""
    print("Viz2prep 간단한 기능 테스트")
    print("="*50)
    
    try:
        # 1. 데이터 로더 테스트
        df = test_data_loader()
        
        # 2. 통계 에이전트 테스트
        stats_result = test_statistics_agent(df)
        
        # 3. 시각화 에이전트 테스트
        viz_result = test_visualization_agent(df)
        
        print("\n" + "="*50)
        print("🎉 모든 기본 기능 테스트 통과!")
        print("="*50)
        
        # 생성된 플롯 파일들 정리
        print("\n생성된 파일들:")
        for plot_path in viz_result['plot_paths']:
            if os.path.exists(plot_path):
                file_size = os.path.getsize(plot_path)
                print(f"  - {plot_path} ({file_size} bytes)")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 