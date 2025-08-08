#!/usr/bin/env python3
"""
실행 및 응답 노드들
전문 코더들이 생성한 코드를 실행하고 최종 응답을 생성하는 노드들
"""

import os
import sys
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflow_state import WorkflowState

def executor_node(state: WorkflowState) -> WorkflowState:
    """
    코드 실행 노드
    
    🔍 INPUT STATES:
    - dataframe: 원본 데이터프레임
    - numeric_code: 수치형 전처리 코드 (1순위)
    - category_code: 범주형 전처리 코드 (2순위)
    - outlier_code: 이상치 전처리 코드 (3순위)
    - nulldata_code: 결측값 전처리 코드 (4순위)
    - corr_code: 상관관계 전처리 코드 (5순위)
    
    📊 OUTPUT STATES:
    - processed_dataframe: 최종 전처리된 데이터프레임
    - execution_results: 각 전처리 단계별 실행 결과
    - execution_errors: 실행 중 발생한 오류들
    - needs_debugging: 디버깅이 필요한지 여부
    - debug_info: 디버깅을 위한 정보
    
    ➡️ NEXT EDGE: responder (성공 시) 또는 debug_router (오류 시)
    """
    print("🔧 [EXEC] 전처리 코드 실행 시작...")
    
    try:
        # 원본 데이터프레임 복사
        original_df = state["dataframe"]
        current_df = original_df.copy()
        
        # 실행 결과 추적
        execution_results = []
        execution_errors = []
        needs_debugging = False
        debug_info = {}
        
        # 실행 순서 정의 (우선순위 기반)
        execution_order = [
            ("nulldata", "nulldata_code", "preprocess_missing_data", "결측값 처리"),
            ("outlier", "outlier_code", "preprocess_outliers", "이상치 처리"),
            ("numeric", "numeric_code", "preprocess_numeric_data", "수치형 전처리"),
            ("category", "category_code", "preprocess_categorical_data", "범주형 전처리"),
            ("correlation", "corr_code", "preprocess_correlation_features", "상관관계 전처리")
        ]
        
        print("📊 [EXEC] 전처리 전 데이터프레임 상태:")
        print(f"   - 크기: {current_df.shape}")
        print(f"   - 결측값: {current_df.isnull().sum().sum()}개")
        print(f"   - 데이터 타입: {dict(current_df.dtypes.value_counts())}")
        
        # 수정된 코드가 있는지 확인
        fixed_code = state.get("fixed_preprocessing_code", "")
        if fixed_code and fixed_code.strip():
            print("🔧 [EXEC] 수정된 코드를 사용하여 재실행...")
            # 수정된 코드를 모든 단계에 적용
            for step, code_key, function_name, description in execution_order:
                state[code_key] = fixed_code
            # 디버깅 완료 후에는 수정된 코드를 초기화하여 무한 루프 방지
            state["fixed_preprocessing_code"] = ""
            # 디버깅 상태도 초기화
            state["debug_status"] = ""
            state["debug_message"] = ""
        
        # 각 코드를 순서대로 실행
        for step, code_key, function_name, description in execution_order:
            print(f"\n🔄 [EXEC] {description} 실행 중...")
            
            code = state.get(code_key, "")
            if not code or code.strip() == "":
                print(f"⏭️  [EXEC] {description} 코드가 없어서 건너뜀")
                continue
            
            try:
                # 전처리 전 상태 기록
                before_shape = current_df.shape
                before_nulls = current_df.isnull().sum().sum()
                
                # 로컬 환경에서 코드 실행
                local_vars = {
                    'df': current_df.copy(),
                    'pd': pd,
                    'np': np,
                    'current_df': current_df.copy()
                }
                
                # 필요한 라이브러리들 미리 import
                exec_globals = {
                    '__builtins__': __builtins__,
                    'pd': pd,
                    'np': np,
                    'pandas': pd,
                    'numpy': np
                }
                
                # sklearn 라이브러리들 import
                try:
                    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
                    from sklearn.ensemble import IsolationForest
                    from sklearn.feature_selection import VarianceThreshold
                    exec_globals.update({
                        'StandardScaler': StandardScaler,
                        'MinMaxScaler': MinMaxScaler,
                        'RobustScaler': RobustScaler,
                        'LabelEncoder': LabelEncoder,
                        'IsolationForest': IsolationForest,
                        'VarianceThreshold': VarianceThreshold
                    })
                except ImportError as e:
                    print(f"⚠️  [EXEC] sklearn import 경고: {e}")
                
                # 코드 실행
                exec(code, exec_globals, local_vars)
                
                # 함수 실행
                if function_name in local_vars:
                    try:
                        processed_df = local_vars[function_name](current_df)
                        if processed_df is not None:
                            current_df = processed_df
                        else:
                            print(f"⚠️  [EXEC] {function_name} 함수가 None을 반환함")
                            continue
                    except Exception as e:
                        print(f"⚠️  [EXEC] {function_name} 함수 실행 오류: {e}")
                        continue
                else:
                    # 함수가 정의되지 않은 경우, df 변수 사용
                    if 'df' in local_vars and local_vars['df'] is not None:
                        current_df = local_vars['df']
                    else:
                        print(f"⚠️  [EXEC] {function_name} 함수를 찾을 수 없음")
                        continue
                
                # 전처리 후 상태 기록
                after_shape = current_df.shape
                after_nulls = current_df.isnull().sum().sum()
                
                # 실행 결과 저장
                step_result = {
                    "step": step,
                    "description": description,
                    "before_shape": before_shape,
                    "after_shape": after_shape,
                    "before_nulls": before_nulls,
                    "after_nulls": after_nulls,
                    "shape_change": f"{before_shape} → {after_shape}",
                    "nulls_change": f"{before_nulls} → {after_nulls}",
                    "status": "success"
                }
                execution_results.append(step_result)
                
                print(f"✅ [EXEC] {description} 완료:")
                print(f"   - 크기 변화: {before_shape} → {after_shape}")
                print(f"   - 결측값 변화: {before_nulls} → {after_nulls}")
                
            except Exception as e:
                error_msg = f"{description} 실행 오류: {str(e)}"
                execution_errors.append(error_msg)
                print(f"❌ [EXEC] {error_msg}")
                
                # 디버깅이 필요한 경우 정보 수집
                needs_debugging = True
                debug_info = {
                    "error_message": error_msg,
                    "original_code": code,
                    "dataframe_info": {
                        "shape": current_df.shape,
                        "columns": list(current_df.columns),
                        "dtypes": dict(current_df.dtypes),
                        "null_counts": dict(current_df.isnull().sum())
                    },
                    "preprocessing_plan": {
                        "step": step,
                        "description": description,
                        "function_name": function_name
                    }
                }
                
                # 실행 실패 결과 저장
                step_result = {
                    "step": step,
                    "description": description,
                    "before_shape": before_shape,
                    "after_shape": before_shape,  # 변화 없음
                    "before_nulls": before_nulls,
                    "after_nulls": before_nulls,  # 변화 없음
                    "shape_change": "변화 없음 (오류)",
                    "nulls_change": "변화 없음 (오류)",
                    "status": "error",
                    "error": str(e)
                }
                execution_results.append(step_result)
                
                # 오류 발생 시 디버깅 정보 수집 (계속 진행)
                print(f"⚠️  [EXEC] {description} 오류 발생, 계속 진행...")
        
        # 모든 전처리 단계 완료 후 오류 확인
        successful_steps = [r for r in execution_results if r.get("status") == "success"]
        failed_steps = [r for r in execution_results if r.get("status") == "error"]
        
        print(f"\n📊 [EXEC] 전처리 완료 요약:")
        print(f"   - 성공한 단계: {len(successful_steps)}/{len(execution_results)}개")
        print(f"   - 실패한 단계: {len(failed_steps)}개")
        
        if failed_steps:
            print(f"🔧 [EXEC] {len(failed_steps)}개 단계에서 오류 발생. Debug Agent로 전달...")
            # 가장 최근 오류 정보 사용
            latest_error = failed_steps[-1]
            return {
                **state,
                "processed_dataframe": current_df,
                "execution_results": execution_results,
                "execution_errors": execution_errors,
                "error_message": latest_error.get("error", "알 수 없는 오류"),
                "preprocessing_code": state.get("numeric_code", "") + "\n" + state.get("category_code", "") + "\n" + state.get("outlier_code", "") + "\n" + state.get("nulldata_code", "") + "\n" + state.get("corr_code", ""),
                "execution_result": {"status": "error"}
            }
        else:
            print(f"\n🎉 [EXEC] 전체 전처리 완료!")
            print(f"📊 [EXEC] 최종 데이터프레임 상태:")
            print(f"   - 최종 크기: {current_df.shape}")
            print(f"   - 최종 결측값: {current_df.isnull().sum().sum()}개")
            print(f"   - 성공한 단계: {len(successful_steps)}/{len(execution_results)}개")
            
            return {
                **state,
                "processed_dataframe": current_df,
                "execution_results": execution_results,
                "execution_errors": execution_errors,
                "execution_result": {"status": "success"}
            }
        
    except Exception as e:
        print(f"❌ [EXEC] 전체 실행 오류: {e}")
        return {
            **state,
            "processed_dataframe": original_df,  # 원본 반환
            "execution_results": [{
                "step": "전체",
                "description": "전체 실행",
                "status": "error",
                "error": str(e)
            }],
            "execution_errors": [f"전체 실행 오류: {str(e)}"],
            "error_message": f"전체 실행 오류: {str(e)}",
            "preprocessing_code": "전체 실행 실패",
            "execution_result": {"status": "error"}
        }


def responder_node(state: WorkflowState) -> WorkflowState:
    """
    최종 응답 노드 - 전처리된 데이터프레임 반환
    
    🔍 INPUT STATES:
    - query: 사용자 요청
    - processed_dataframe: 최종 전처리된 데이터프레임
    - execution_results: 각 전처리 단계별 실행 결과
    - execution_errors: 실행 중 발생한 오류들
    
    📊 OUTPUT STATES:
    - final_response: 전처리된 데이터프레임과 기본 정보
    
    ➡️ NEXT EDGE: END
    """
    print("📝 [RESPONSE] 최종 응답 생성 중...")
    
    try:
        query = state.get("query", "")
        processed_df = state.get("processed_dataframe")
        execution_results = state.get("execution_results", [])
        execution_errors = state.get("execution_errors", [])
        original_df = state.get("dataframe")
        
        # 디버깅 정보 확인
        debug_status = state.get("debug_status", "")
        debug_message = state.get("debug_message", "")
        fixed_preprocessing_code = state.get("fixed_preprocessing_code", "")
        debug_analysis = state.get("debug_analysis", {})
        
        # 실행 결과 요약
        successful_steps = [r for r in execution_results if r.get("status") == "success"]
        failed_steps = [r for r in execution_results if r.get("status") == "error"]
        
        # 데이터프레임 변화 요약
        original_shape = original_df.shape if original_df is not None else (0, 0)
        final_shape = processed_df.shape if processed_df is not None else (0, 0)
        original_nulls = original_df.isnull().sum().sum() if original_df is not None else 0
        final_nulls = processed_df.isnull().sum().sum() if processed_df is not None else 0
        
        # 전처리 요약 정보 생성
        preprocessing_summary = {
            "user_query": query,
            "original_data_info": {
                "shape": original_shape,
                "missing_values": original_nulls,
                "data_types": dict(original_df.dtypes.value_counts()) if original_df is not None else {}
            },
            "final_data_info": {
                "shape": final_shape,
                "missing_values": final_nulls,
                "data_types": dict(processed_df.dtypes.value_counts()) if processed_df is not None else {}
            },
            "processing_steps": {
                "total_steps": len(execution_results),
                "successful_steps": len(successful_steps),
                "failed_steps": len(failed_steps),
                "success_rate": len(successful_steps) / len(execution_results) * 100 if execution_results else 0
            },
            "data_quality_improvements": {
                "missing_values_removed": original_nulls - final_nulls,
                "shape_change": f"{original_shape} → {final_shape}",
                "data_completeness": (1 - final_nulls / (final_shape[0] * final_shape[1])) * 100 if final_shape[0] * final_shape[1] > 0 else 0
            },
            "execution_details": execution_results,
            "errors": execution_errors,
            "debug_info": {
                "debug_status": debug_status,
                "debug_message": debug_message,
                "debug_analysis": debug_analysis,
                "has_fixed_code": bool(fixed_preprocessing_code)
            }
        }
        
        # 간단한 성공 메시지 생성
        success_message = f"""
✅ 전처리 완료!

📊 데이터 정보:
- 원본: {original_shape[0]}행 x {original_shape[1]}열
- 최종: {final_shape[0]}행 x {final_shape[1]}열
- 성공률: {preprocessing_summary['processing_steps']['success_rate']:.1f}%
- 결측값 변화: {original_nulls} → {final_nulls}

🎯 전처리된 데이터프레임이 준비되었습니다. ML 학습에 바로 사용하세요!
"""
        
        print("✅ [RESPONSE] 최종 응답 생성 완료")
        print(f"📊 [RESPONSE] 요약:")
        print(f"   - 성공률: {preprocessing_summary['processing_steps']['success_rate']:.1f}%")
        print(f"   - 데이터 완전성: {preprocessing_summary['data_quality_improvements']['data_completeness']:.1f}%")
        print(f"   - 제거된 결측값: {preprocessing_summary['data_quality_improvements']['missing_values_removed']}개")
        
        return {
            **state,
            "final_response": success_message,
            "preprocessing_summary": preprocessing_summary
        }
        
    except Exception as e:
        print(f"❌ [RESPONSE] 응답 생성 오류: {e}")
        
        # 기본 응답 생성
        basic_summary = {
            "user_query": query,
            "status": "error",
            "error_message": str(e),
            "processing_steps": {"total_steps": 0, "successful_steps": 0, "failed_steps": 0}
        }
        
        error_message = f"""
⚠️ 전처리 완료 (오류 발생)

📊 데이터 정보:
- 원본: {original_df.shape if original_df is not None else (0, 0)}
- 최종: {processed_df.shape if processed_df is not None else (0, 0)}

❌ 응답 생성 중 오류: {str(e)}

🎯 전처리된 데이터프레임은 사용 가능합니다.
"""
        
        return {
            **state,
            "final_response": error_message,
            "preprocessing_summary": basic_summary
        } 