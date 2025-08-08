#!/usr/bin/env python3
"""
디버깅 기능이 포함된 워크플로우 테스트
"""

import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from workflow_graph import build_specialized_workflow
from workflow_state import WorkflowState

def test_iris_workflow():
    """Iris 데이터셋으로 워크플로우 테스트"""
    print("🌺 [TEST] Iris 데이터셋으로 워크플로우 테스트 시작")
    
    # Iris 데이터 로드
    try:
        df = pd.read_csv("datasets/Iris/Iris.csv")
        print(f"✅ [TEST] Iris 데이터 로드 완료: {df.shape}")
        print(f"📊 [TEST] 컬럼: {list(df.columns)}")
        print(f"📊 [TEST] 데이터 타입: {df.dtypes.to_dict()}")
    except Exception as e:
        print(f"❌ [TEST] Iris 데이터 로드 실패: {e}")
        return
    
    # 워크플로우 상태 초기화
    initial_state = WorkflowState(
        dataframe=df,
        query="Iris 데이터셋의 수치형 변수들을 정규화하고, 범주형 변수를 인코딩해주세요. 이상치도 처리해주세요.",
        numeric_eda_result={},
        category_eda_result={},
        outlier_eda_result={},
        nulldata_eda_result={},
        corr_eda_result={},
        numeric_plan={},
        category_plan={},
        outlier_plan={},
        nulldata_plan={},
        corr_plan={},
        numeric_code="",
        category_code="",
        outlier_code="",
        nulldata_code="",
        corr_code="",
        preprocessing_result={},
        final_response="",
        debug_status="",
        debug_message="",
        fixed_preprocessing_code="",
        debug_analysis=""
    )
    
    # 워크플로우 그래프 생성
    workflow = build_specialized_workflow()
    
    print("🚀 [TEST] 워크플로우 실행 시작...")
    
    try:
        # 워크플로우 실행
        result = workflow.invoke(initial_state)
        
        print("\n" + "="*60)
        print("🎉 [TEST] 워크플로우 실행 완료!")
        print("="*60)
        
        # 결과 출력
        print(f"📊 [RESULT] 최종 응답:")
        print(result.get("final_response", "응답 없음"))
        
        # 전처리된 데이터프레임 정보 출력
        processed_df = result.get("processed_dataframe")
        if processed_df is not None:
            print(f"\n📈 [RESULT] 전처리된 데이터프레임:")
            print(f"   - 크기: {processed_df.shape}")
            print(f"   - 컬럼: {list(processed_df.columns)}")
            print(f"   - 결측값: {processed_df.isnull().sum().sum()}개")
            print(f"   - 데이터 타입: {processed_df.dtypes.to_dict()}")
            
            # 처음 5행 출력
            print(f"\n📋 [RESULT] 전처리된 데이터 샘플 (처음 5행):")
            print(processed_df.head())
        
        # 디버깅 정보가 있다면 출력
        if result.get("debug_status"):
            print(f"\n🔧 [DEBUG] 디버깅 상태: {result.get('debug_status')}")
            print(f"🔧 [DEBUG] 디버깅 메시지: {result.get('debug_message')}")
            print(f"🔧 [DEBUG] 디버깅 분석: {result.get('debug_analysis')}")
        
        # 생성된 코드들 출력
        if result.get("numeric_code"):
            print(f"\n💻 [CODE] 수치형 전처리 코드:")
            print(result.get("numeric_code"))
        
        if result.get("category_code"):
            print(f"\n💻 [CODE] 범주형 전처리 코드:")
            print(result.get("category_code"))
        
        if result.get("outlier_code"):
            print(f"\n💻 [CODE] 이상치 전처리 코드:")
            print(result.get("outlier_code"))
        
        if result.get("nulldata_code"):
            print(f"\n💻 [CODE] 결측값 전처리 코드:")
            print(result.get("nulldata_code"))
        
        if result.get("corr_code"):
            print(f"\n💻 [CODE] 상관관계 전처리 코드:")
            print(result.get("corr_code"))
        
    except Exception as e:
        print(f"❌ [TEST] 워크플로우 실행 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_iris_workflow() 