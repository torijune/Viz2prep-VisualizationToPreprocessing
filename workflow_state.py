#!/usr/bin/env python3
"""
LangGraph 워크플로우 State 정의
전체 워크플로우에서 사용되는 모든 state들을 정의하고 각 노드별 입출력을 명시합니다.
"""

from typing import Annotated, TypedDict, List, Dict, Any, Optional
import pandas as pd

class WorkflowState(TypedDict):
    """
    전체 워크플로우 State 정의
    
    ===================
    📊 INPUT STATES (사용자로부터 입력)
    ===================
    - query: 사용자의 전처리 요청
    - X: 특성 변수 데이터프레임
    - Y: 타겟 변수 데이터프레임
    
    ===================
    🔍 EDA STATES (EDA 에이전트들이 생성)
    ===================
    - numeric_eda_result: 수치형 변수 EDA 결과
    - category_eda_result: 범주형 변수 EDA 결과  
    - outlier_eda_result: 이상치 EDA 결과
    - nulldata_eda_result: 결측값 EDA 결과
    - corr_eda_result: 상관관계 EDA 결과
    
    ===================
    📋 PLANNING STATES (전문 플래너들이 생성)
    ===================
    - numeric_plan: 수치형 전처리 계획
    - category_plan: 범주형 전처리 계획
    - outlier_plan: 이상치 전처리 계획
    - nulldata_plan: 결측값 전처리 계획
    - corr_plan: 상관관계 전처리 계획
    
    ===================
    💻 CODING STATES (전문 코더들이 생성)
    ===================
    - numeric_code: 수치형 전처리 코드
    - category_code: 범주형 전처리 코드
    - outlier_code: 이상치 전처리 코드
    - nulldata_code: 결측값 전처리 코드
    - corr_code: 상관관계 전처리 코드
    
    ===================
    🔧 EXECUTION STATES (코드 실행 결과)
    ===================
    - X_processed: 전처리된 특성 변수 데이터프레임
    - Y_processed: 전처리된 타겟 변수 데이터프레임
    - execution_results: 각 전처리 단계별 실행 결과
    - execution_errors: 실행 중 발생한 오류들
    
    ===================
    ✅ OUTPUT STATES (최종 결과)
    ===================
    - final_answer: 최종 응답
    - preprocessing_summary: 전처리 요약 보고서
    """
    
    # 📊 INPUT STATES
    query: Annotated[str, "사용자의 전처리 요청 쿼리"]
    X: Annotated[pd.DataFrame, "특성 변수 데이터프레임"]
    Y: Annotated[pd.DataFrame, "타겟 변수 데이터프레임"]
    
    # 🔍 EDA STATES
    numeric_eda_result: Annotated[Dict[str, Any], "수치형 변수 EDA 결과 (numeric_agent → numeric_planner)"]
    category_eda_result: Annotated[Dict[str, Any], "범주형 변수 EDA 결과 (category_agent → category_planner)"]
    outlier_eda_result: Annotated[Dict[str, Any], "이상치 EDA 결과 (outlier_agent → outlier_planner)"]
    nulldata_eda_result: Annotated[Dict[str, Any], "결측값 EDA 결과 (nulldata_agent → nulldata_planner)"]
    corr_eda_result: Annotated[Dict[str, Any], "상관관계 EDA 결과 (corr_agent → corr_planner)"]
    
    # 📋 PLANNING STATES
    numeric_plan: Annotated[Dict[str, Any], "수치형 전처리 계획 (numeric_planner → numeric_coder)"]
    category_plan: Annotated[Dict[str, Any], "범주형 전처리 계획 (category_planner → category_coder)"]
    outlier_plan: Annotated[Dict[str, Any], "이상치 전처리 계획 (outlier_planner → outlier_coder)"]
    nulldata_plan: Annotated[Dict[str, Any], "결측값 전처리 계획 (nulldata_planner → nulldata_coder)"]
    corr_plan: Annotated[Dict[str, Any], "상관관계 전처리 계획 (corr_planner → corr_coder)"]
    
    # 💻 CODING STATES
    numeric_code: Annotated[str, "수치형 전처리 코드 (numeric_coder → executor)"]
    category_code: Annotated[str, "범주형 전처리 코드 (category_coder → executor)"]
    outlier_code: Annotated[str, "이상치 전처리 코드 (outlier_coder → executor)"]
    nulldata_code: Annotated[str, "결측값 전처리 코드 (nulldata_coder → executor)"]
    corr_code: Annotated[str, "상관관계 전처리 코드 (corr_coder → executor)"]
    
    # 🔧 EXECUTION STATES
    X_processed: Annotated[pd.DataFrame, "전처리된 특성 변수 데이터프레임 (executor → debug_agent)"]
    Y_processed: Annotated[pd.DataFrame, "전처리된 타겟 변수 데이터프레임 (executor → debug_agent)"]
    execution_results: Annotated[List[Dict[str, Any]], "각 전처리 단계별 실행 결과 (executor → debug_agent)"]
    execution_errors: Annotated[List[str], "실행 중 발생한 오류들 (executor → debug_agent)"]
    
    # 🐛 DEBUG STATES
    debug_status: Annotated[str, "디버깅 상태 (debug_agent → executor/responder)"]
    debug_message: Annotated[str, "디버깅 메시지 (debug_agent → executor/responder)"]
    fixed_preprocessing_code: Annotated[str, "수정된 전처리 코드 (debug_agent → executor)"]
    debug_analysis: Annotated[Dict[str, Any], "디버깅 분석 결과 (debug_agent → executor/responder)"]
    
    # ✅ OUTPUT STATES
    final_answer: Annotated[str, "최종 응답 (responder → END)"]
    preprocessing_summary: Annotated[Dict[str, Any], "전처리 요약 보고서 (responder → END)"]


class NodeIOMapping:
    """
    각 노드별 입력/출력 State 매핑
    디버깅과 워크플로우 이해를 위한 참조용
    """
    
    EDA_AGENTS = {
        "numeric_agent": {
            "inputs": ["X", "Y"],
            "outputs": ["numeric_eda_result"],
            "next_edge": "numeric_planner"
        },
        "category_agent": {
            "inputs": ["X", "Y"],
            "outputs": ["category_eda_result"],
            "next_edge": "category_planner"
        },
        "outlier_agent": {
            "inputs": ["X", "Y"],
            "outputs": ["outlier_eda_result"],
            "next_edge": "outlier_planner"
        },
        "nulldata_agent": {
            "inputs": ["X", "Y"],
            "outputs": ["nulldata_eda_result"],
            "next_edge": "nulldata_planner"
        },
        "corr_agent": {
            "inputs": ["X", "Y"],
            "outputs": ["corr_eda_result"],
            "next_edge": "corr_planner"
        }
    }
    
    PLANNING_AGENTS = {
        "numeric_planner": {
            "inputs": ["query", "numeric_eda_result"],
            "outputs": ["numeric_plan"],
            "next_edge": "numeric_coder"
        },
        "category_planner": {
            "inputs": ["query", "category_eda_result"],
            "outputs": ["category_plan"],
            "next_edge": "category_coder"
        },
        "outlier_planner": {
            "inputs": ["query", "outlier_eda_result"],
            "outputs": ["outlier_plan"],
            "next_edge": "outlier_coder"
        },
        "nulldata_planner": {
            "inputs": ["query", "nulldata_eda_result"],
            "outputs": ["nulldata_plan"],
            "next_edge": "nulldata_coder"
        },
        "corr_planner": {
            "inputs": ["query", "corr_eda_result"],
            "outputs": ["corr_plan"],
            "next_edge": "corr_coder"
        }
    }
    
    CODING_AGENTS = {
        "numeric_coder": {
            "inputs": ["numeric_plan", "numeric_eda_result"],
            "outputs": ["numeric_code"],
            "next_edge": "executor (순서: 1번째)"
        },
        "category_coder": {
            "inputs": ["category_plan", "category_eda_result"],
            "outputs": ["category_code"],
            "next_edge": "executor (순서: 2번째)"
        },
        "outlier_coder": {
            "inputs": ["outlier_plan", "outlier_eda_result"],
            "outputs": ["outlier_code"],
            "next_edge": "executor (순서: 3번째)"
        },
        "nulldata_coder": {
            "inputs": ["nulldata_plan", "nulldata_eda_result"],
            "outputs": ["nulldata_code"],
            "next_edge": "executor (순서: 4번째)"
        },
        "corr_coder": {
            "inputs": ["corr_plan", "corr_eda_result"],
            "outputs": ["corr_code"],
            "next_edge": "executor (순서: 5번째)"
        }
    }
    
    EXECUTION_AGENT = {
        "executor": {
            "inputs": ["X", "Y", "numeric_code", "category_code", "outlier_code", "nulldata_code", "corr_code", "fixed_preprocessing_code"],
            "outputs": ["X_processed", "Y_processed", "execution_results", "execution_errors", "error_message", "preprocessing_code", "execution_result"],
            "next_edge": "debug_agent"
        }
    }
    
    DEBUG_AGENT = {
        "debug_agent": {
            "inputs": ["X", "Y", "X_processed", "Y_processed", "execution_results", "execution_errors", "error_message", "preprocessing_code", "execution_result"],
            "outputs": ["debug_status", "debug_message", "fixed_preprocessing_code", "debug_analysis"],
            "next_edge": "executor (성공 시) 또는 responder (실패 시)"
        }
    }
    
    RESPONSE_AGENT = {
        "responder": {
            "inputs": ["query", "X_processed", "Y_processed", "execution_results", "execution_errors", "debug_status", "debug_message", "debug_analysis"],
            "outputs": ["final_answer", "preprocessing_summary"],
            "next_edge": "END"
        }
    }


def print_workflow_state_summary():
    """
    워크플로우 State 요약을 출력합니다.
    """
    print("=" * 80)
    print("📊 WORKFLOW STATE SUMMARY")
    print("=" * 80)
    
    print("\n🔍 EDA AGENTS:")
    for agent, mapping in NodeIOMapping.EDA_AGENTS.items():
        print(f"  {agent}: {mapping['inputs']} → {mapping['outputs']} → {mapping['next_edge']}")
    
    print("\n📋 PLANNING AGENTS:")
    for agent, mapping in NodeIOMapping.PLANNING_AGENTS.items():
        print(f"  {agent}: {mapping['inputs']} → {mapping['outputs']} → {mapping['next_edge']}")
    
    print("\n💻 CODING AGENTS:")
    for agent, mapping in NodeIOMapping.CODING_AGENTS.items():
        print(f"  {agent}: {mapping['inputs']} → {mapping['outputs']} → {mapping['next_edge']}")
    
    print("\n🔧 EXECUTION AGENT:")
    for agent, mapping in NodeIOMapping.EXECUTION_AGENT.items():
        print(f"  {agent}: {mapping['inputs']} → {mapping['outputs']} → {mapping['next_edge']}")
    
    print("\n🐛 DEBUG AGENT:")
    for agent, mapping in NodeIOMapping.DEBUG_AGENT.items():
        print(f"  {agent}: {mapping['inputs']} → {mapping['outputs']} → {mapping['next_edge']}")
    
    print("\n✅ RESPONSE AGENT:")
    for agent, mapping in NodeIOMapping.RESPONSE_AGENT.items():
        print(f"  {agent}: {mapping['inputs']} → {mapping['outputs']} → {mapping['next_edge']}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    print_workflow_state_summary() 