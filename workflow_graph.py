#!/usr/bin/env python3
"""
새로운 LangGraph 워크플로우 - 전문화된 다중 에이전트 시스템
EDA → Planning → Coding → Execution → Response

===========================================
🔄 WORKFLOW ARCHITECTURE
===========================================

📊 EDA LAYER (5 parallel agents):
numeric_agent → category_agent → outlier_agent → nulldata_agent → corr_agent
        ↓              ↓              ↓              ↓              ↓

📋 PLANNING LAYER (5 parallel planners):
numeric_planner → category_planner → outlier_planner → nulldata_planner → corr_planner
        ↓              ↓              ↓              ↓              ↓

💻 CODING LAYER (5 parallel coders):
numeric_coder → category_coder → outlier_coder → nulldata_coder → corr_coder
        ↓              ↓              ↓              ↓              ↓
        
🔧 EXECUTION LAYER (1 executor):
                              executor
                                ↓
                                
📝 RESPONSE LAYER (1 responder):
                             responder
                                ↓
                               END
"""

import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

from langgraph.graph import StateGraph, END
from typing import Dict, Any
import pandas as pd

# 워크플로우 State import
from workflow_state import WorkflowState

# 노드들 import
from agents.eda_nodes import (
    numeric_agent_node,
    category_agent_node,
    outlier_agent_node,
    nulldata_agent_node,
    corr_agent_node
)

from agents.planning_nodes import (
    numeric_planner_node,
    category_planner_node,
    outlier_planner_node,
    nulldata_planner_node,
    corr_planner_node
)

from agents.coding_nodes import (
    numeric_coder_node,
    category_coder_node,
    outlier_coder_node,
    nulldata_coder_node,
    corr_coder_node
)

from agents.execution_response_nodes import (
    executor_node,
    responder_node
)

from agents.debug_agent import debug_agent_node

def check_eda_completion(state: WorkflowState) -> str:
    """
    EDA 완료 확인 조건부 함수
    
    🔍 INPUT STATES: 모든 EDA 결과들
    ➡️ OUTPUT EDGES: 
        - "all_eda_complete" → 모든 플래너들 동시 실행
        - "eda_incomplete" → 오류 처리 (현재는 강제 진행)
    """
    eda_results = [
        state.get("numeric_eda_result", {}),
        state.get("category_eda_result", {}),
        state.get("outlier_eda_result", {}),
        state.get("nulldata_eda_result", {}),
        state.get("corr_eda_result", {})
    ]
    
    # 모든 EDA가 완료되었는지 확인
    completed_count = sum(1 for result in eda_results if result.get("status") == "success")
    
    print(f"🔍 [CONDITION] EDA 완료 확인: {completed_count}/5개 완료")
    
    # 하나라도 완료되면 다음 단계로 진행 (유연한 정책)
    if completed_count > 0:
        return "all_eda_complete"
    else:
        return "eda_incomplete"

def check_planning_completion(state: WorkflowState) -> str:
    """
    Planning 완료 확인 조건부 함수
    
    🔍 INPUT STATES: 모든 Planning 결과들
    ➡️ OUTPUT EDGES:
        - "all_planning_complete" → 모든 코더들 동시 실행
        - "planning_incomplete" → 오류 처리 (현재는 강제 진행)
    """
    planning_results = [
        state.get("numeric_plan", {}),
        state.get("category_plan", {}),
        state.get("outlier_plan", {}),
        state.get("nulldata_plan", {}),
        state.get("corr_plan", {})
    ]
    
    completed_count = sum(1 for result in planning_results if result.get("status") == "success")
    
    print(f"🔍 [CONDITION] Planning 완료 확인: {completed_count}/5개 완료")
    
    if completed_count > 0:
        return "all_planning_complete"
    else:
        return "planning_incomplete"

def check_coding_completion(state: WorkflowState) -> str:
    """
    Coding 완료 확인 조건부 함수
    
    🔍 INPUT STATES: 모든 Coding 결과들
    ➡️ OUTPUT EDGES:
        - "all_coding_complete" → executor 실행
        - "coding_incomplete" → 오류 처리 (현재는 강제 진행)
    """
    coding_results = [
        state.get("numeric_code", ""),
        state.get("category_code", ""),
        state.get("outlier_code", ""),
        state.get("nulldata_code", ""),
        state.get("corr_code", "")
    ]
    
    completed_count = sum(1 for code in coding_results if code and code.strip())
    
    print(f"🔍 [CONDITION] Coding 완료 확인: {completed_count}/5개 완료")
    
    if completed_count > 0:
        return "all_coding_complete"
    else:
        return "coding_incomplete"

def check_debug_completion(state: WorkflowState) -> str:
    """
    디버깅 완료 확인 조건부 함수
    
    🔍 INPUT STATES: 디버깅 결과들
    ➡️ OUTPUT EDGES: 
        - "debug_fixed" → 수정된 코드로 다시 실행
        - "debug_failed" → 응답으로 진행
        - "debug_no_error" → 응답으로 진행
    """
    debug_status = state.get("debug_status", "")
    fixed_preprocessing_code = state.get("fixed_preprocessing_code", "")
    
    print(f"🔍 [CONDITION] 디버깅 완료 확인: {debug_status}")
    
    # 오류가 없는 경우 응답으로 진행
    if debug_status == "no_error":
        print("✅ [CONDITION] 오류 없음, 응답으로 진행")
        return "debug_no_error"
    # 디버깅이 성공하고 수정된 코드가 있는 경우 다시 실행
    elif debug_status == "fixed" and fixed_preprocessing_code:
        print("✅ [CONDITION] 디버깅 성공, 수정된 코드로 다시 실행")
        return "debug_fixed"
    else:
        print("⚠️  [CONDITION] 디버깅 실패 또는 수정된 코드 없음, 응답으로 진행")
        return "debug_failed"

def build_specialized_workflow() -> StateGraph:
    """
    전문화된 다중 에이전트 워크플로우 구축 (디버깅 기능 포함)
    
    ===========================================
    📊 NODE REGISTRATION (18개 노드)
    ===========================================
    
    🔍 EDA NODES (5개):
    - numeric_agent: 수치형 변수 EDA
    - category_agent: 범주형 변수 EDA  
    - outlier_agent: 이상치 EDA
    - nulldata_agent: 결측값 EDA
    - corr_agent: 상관관계 EDA
    
    📋 PLANNING NODES (5개):
    - numeric_planner: 수치형 전처리 계획
    - category_planner: 범주형 전처리 계획
    - outlier_planner: 이상치 전처리 계획  
    - nulldata_planner: 결측값 전처리 계획
    - corr_planner: 상관관계 전처리 계획
    
    💻 CODING NODES (5개):
    - numeric_coder: 수치형 전처리 코드 생성
    - category_coder: 범주형 전처리 코드 생성
    - outlier_coder: 이상치 전처리 코드 생성
    - nulldata_coder: 결측값 전처리 코드 생성
    - corr_coder: 상관관계 전처리 코드 생성
    
    🔧 EXECUTION NODE (1개):
    - executor: 모든 코드 통합 실행
    
    🐛 DEBUG NODE (1개):
    - debug_agent: 코드 오류 디버깅 및 수정
    
    📝 RESPONSE NODE (1개):
    - responder: 최종 응답 생성
    
    ===========================================
    🔗 EDGE CONNECTIONS
    ===========================================
    
    ▶️ ENTRY POINT: START → 모든 EDA 노드들 (parallel)
    
    🔄 LAYER 1 (EDA): 
    - START → [numeric_agent, category_agent, outlier_agent, nulldata_agent, corr_agent]
    
    🔀 CONDITION 1: EDA 완료 확인
    - [all EDA nodes] → eda_completion_check → [all Planning nodes]
    
    🔄 LAYER 2 (PLANNING):
    - numeric_agent → numeric_planner
    - category_agent → category_planner  
    - outlier_agent → outlier_planner
    - nulldata_agent → nulldata_planner
    - corr_agent → corr_planner
    
    🔀 CONDITION 2: Planning 완료 확인
    - [all Planning nodes] → planning_completion_check → [all Coding nodes]
    
    🔄 LAYER 3 (CODING):
    - numeric_planner → numeric_coder
    - category_planner → category_coder
    - outlier_planner → outlier_coder  
    - nulldata_planner → nulldata_coder
    - corr_planner → corr_coder
    
    🔀 CONDITION 3: Coding 완료 확인  
    - [all Coding nodes] → coding_completion_check → executor
    
    🔄 LAYER 4 (EXECUTION):
    - [all Coding nodes] → executor
    
    🔄 LAYER 5 (DEBUG):
    - executor → debug_agent → 조건부 라우팅
    
    🔄 LAYER 6 (RESPONSE):
    - debug_agent → responder → END (디버깅 실패 시)
    - debug_agent → executor → responder → END (디버깅 성공 시)
    
    """
    
    print("🏗️  [GRAPH] 전문화된 워크플로우 그래프 구축 시작...")
    
    # StateGraph 생성
    builder = StateGraph(state_schema=WorkflowState)
    
    # ===========================================
    # 📊 EDA NODES 등록 (5개)
    # ===========================================
    print("📊 [GRAPH] EDA 노드들 등록 중...")
    builder.add_node("numeric_agent", numeric_agent_node)           # 수치형 EDA
    builder.add_node("category_agent", category_agent_node)         # 범주형 EDA  
    builder.add_node("outlier_agent", outlier_agent_node)           # 이상치 EDA
    builder.add_node("nulldata_agent", nulldata_agent_node)         # 결측값 EDA
    builder.add_node("corr_agent", corr_agent_node)                 # 상관관계 EDA
    
    # ===========================================
    # 📋 PLANNING NODES 등록 (5개)
    # ===========================================
    print("📋 [GRAPH] Planning 노드들 등록 중...")
    builder.add_node("numeric_planner", numeric_planner_node)       # 수치형 Planning
    builder.add_node("category_planner", category_planner_node)     # 범주형 Planning
    builder.add_node("outlier_planner", outlier_planner_node)       # 이상치 Planning
    builder.add_node("nulldata_planner", nulldata_planner_node)     # 결측값 Planning
    builder.add_node("corr_planner", corr_planner_node)             # 상관관계 Planning
    
    # ===========================================
    # 💻 CODING NODES 등록 (5개)
    # ===========================================
    print("💻 [GRAPH] Coding 노드들 등록 중...")
    builder.add_node("numeric_coder", numeric_coder_node)           # 수치형 Coding
    builder.add_node("category_coder", category_coder_node)         # 범주형 Coding
    builder.add_node("outlier_coder", outlier_coder_node)           # 이상치 Coding  
    builder.add_node("nulldata_coder", nulldata_coder_node)         # 결측값 Coding
    builder.add_node("corr_coder", corr_coder_node)                 # 상관관계 Coding
    
    # ===========================================
    # 🔧 EXECUTION & DEBUG & RESPONSE NODES 등록 (3개)
    # ===========================================
    print("🔧 [GRAPH] Execution, Debug & Response 노드들 등록 중...")
    builder.add_node("executor", executor_node)                     # 코드 실행
    builder.add_node("debug_agent", debug_agent_node)              # 코드 디버깅
    builder.add_node("responder", responder_node)                   # 최종 응답
    
    # ===========================================
    # ▶️ ENTRY POINT 설정
    # ===========================================
    print("▶️ [GRAPH] Entry Point 설정 중...")
    # START에서 모든 EDA 노드들로 동시 시작
    builder.set_entry_point("numeric_agent")
    
    # ===========================================
    # 🔗 EDA LAYER EDGES (병렬 실행)
    # ===========================================
    print("🔗 [GRAPH] EDA Layer 엣지 연결 중...")
    # 다른 EDA 노드들도 START에서 시작
    builder.add_edge("numeric_agent", "category_agent")
    builder.add_edge("category_agent", "outlier_agent") 
    builder.add_edge("outlier_agent", "nulldata_agent")
    builder.add_edge("nulldata_agent", "corr_agent")
    
    # ===========================================
    # 🔗 EDA → PLANNING LAYER EDGES
    # ===========================================
    print("🔗 [GRAPH] EDA → Planning Layer 엣지 연결 중...")
    # EDA 완료 후 해당 도메인 플래너로 연결
    builder.add_edge("corr_agent", "numeric_planner")      # 마지막 EDA 완료 후 Planning 시작
    builder.add_edge("numeric_planner", "category_planner")
    builder.add_edge("category_planner", "outlier_planner")
    builder.add_edge("outlier_planner", "nulldata_planner") 
    builder.add_edge("nulldata_planner", "corr_planner")
    
    # ===========================================
    # 🔗 PLANNING → CODING LAYER EDGES
    # ===========================================
    print("🔗 [GRAPH] Planning → Coding Layer 엣지 연결 중...")
    # Planning 완료 후 해당 도메인 코더로 연결
    builder.add_edge("corr_planner", "numeric_coder")      # 마지막 Planning 완료 후 Coding 시작
    builder.add_edge("numeric_coder", "category_coder")
    builder.add_edge("category_coder", "outlier_coder")
    builder.add_edge("outlier_coder", "nulldata_coder")
    builder.add_edge("nulldata_coder", "corr_coder")
    
    # ===========================================
    # 🔗 CODING → EXECUTION → DEBUG → RESPONSE EDGES
    # ===========================================
    print("🔗 [GRAPH] Coding → Execution → Debug → Response 엣지 연결 중...")
    # 모든 코딩 완료 후 실행으로 연결
    builder.add_edge("corr_coder", "executor")             # 마지막 Coding 완료 후 Execution
    
    # 실행 완료 후 디버깅으로 연결
    builder.add_edge("executor", "debug_agent")            # Execution → Debug
    
    # 디버깅 완료 후 조건부 라우팅
    builder.add_conditional_edges(
        "debug_agent",
        check_debug_completion,
        {
            "debug_fixed": "executor",      # 수정된 코드로 다시 실행
            "debug_failed": "responder",    # 응답으로 진행
            "debug_no_error": "responder"   # 오류 없음, 응답으로 진행
        }
    )
    
    # 응답 완료 후 종료
    builder.add_edge("responder", END)                     # Response → END
    
    print("✅ [GRAPH] 워크플로우 그래프 구축 완료!")
    print("📊 [GRAPH] 총 노드 수: 18개 (EDA 5개 + Planning 5개 + Coding 5개 + Execution 1개 + Debug 1개 + Response 1개)")
    print("🔗 [GRAPH] 총 엣지 수: 17개")
    
    return builder.compile()

def run_workflow_demo():
    """워크플로우 데모 실행 - Iris 데이터셋 사용"""
    print("=" * 80)
    print("🌺 Iris 데이터셋으로 전문화된 다중 에이전트 워크플로우 데모 시작")
    print("=" * 80)
    
    # Iris 데이터셋 로드
    try:
        iris_path = "datasets/Iris/Iris.csv"
        df = pd.read_csv(iris_path)
        print(f"✅ [TEST] Iris 데이터 로드 완료: {df.shape}")
        print(f"📊 [TEST] 컬럼: {list(df.columns)}")
        print(f"📊 [TEST] 데이터 타입: {df.dtypes.to_dict()}")
    except FileNotFoundError:
        print(f"❌ [ERROR] Iris 데이터셋을 찾을 수 없습니다: {iris_path}")
        print("📁 [INFO] datasets/Iris/Iris.csv 파일이 존재하는지 확인해주세요.")
        return
    except Exception as e:
        print(f"❌ [ERROR] 데이터 로드 오류: {e}")
        return
    
    # 워크플로우 그래프 구축
    workflow = build_specialized_workflow()
    
    # 초기 상태 설정
    initial_state = {
        "query": "Iris 데이터셋을 머신러닝 모델 학습에 적합하도록 전처리해주세요. Species 변수를 원핫 인코딩하고 수치형 변수들을 정규화해주세요.",
        "dataframe": df
    }
    
    print("\n🔄 워크플로우 실행 시작...")
    print("=" * 50)
    
    # 워크플로우 실행
    try:
        result = workflow.invoke(initial_state)
        
        print("\n" + "=" * 80)
        print("🎉 워크플로우 실행 완료!")
        print("=" * 80)
        
        # 결과 출력
        print("\n📝 최종 응답:")
        print(result.get("final_answer", "응답을 생성할 수 없습니다."))
        
        print("\n📊 전처리 요약:")
        summary = result.get("preprocessing_summary", {})
        if summary:
            print(f"   - 성공률: {summary.get('processing_steps', {}).get('success_rate', 0):.1f}%")
            print(f"   - 데이터 완전성: {summary.get('data_quality_improvements', {}).get('data_completeness', 0):.1f}%")
            print(f"   - 제거된 결측값: {summary.get('data_quality_improvements', {}).get('missing_values_removed', 0)}개")
        
        # 처리된 데이터프레임 정보
        processed_df = result.get("processed_dataframe")
        if processed_df is not None:
            print(f"\n📈 데이터프레임 변화:")
            print(f"   - 원본: {df.shape} → 최종: {processed_df.shape}")
            print(f"   - 결측값: {df.isnull().sum().sum()} → {processed_df.isnull().sum().sum()}")
            
            print(f"\n📋 처리된 데이터 샘플:")
            print(processed_df.head(3))
        
    except Exception as e:
        print(f"❌ 워크플로우 실행 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # State 요약 출력
    from workflow_state import print_workflow_state_summary
    print_workflow_state_summary()
    
    print("\n")
    
    # 워크플로우 데모 실행
    run_workflow_demo() 