"""
LangGraph를 사용하여 멀티모달 LLM 에이전트 그래프를 구축합니다.
"""

from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import pandas as pd

# 에이전트들 import
from agents.data_loader import data_loader
from agents.text_analysis_agent import text_analysis_agent
from agents.visualization_agent import visualization_agent
from agents.domain_analysis import domain_analysis_agent
from agents.preprocessing_agent import preprocessing_agent
from agents.evaluator import evaluator


def create_agent_graph():
    """
    멀티모달 LLM 에이전트 그래프를 생성합니다.
    
    Returns:
        구성된 StateGraph
    """
    
    # 상태 그래프 생성
    workflow = StateGraph(Dict[str, Any])
    
    # 노드 추가
    ## 데이터프레임 형태로 state 전달
    workflow.add_node("data_loader", data_loader)
    workflow.add_node("text_analysis_agent", text_analysis_agent)
    workflow.add_node("visualization_agent", visualization_agent)
    workflow.add_node("domain_analysis_agent", domain_analysis_agent)
    workflow.add_node("preprocessing_agent", preprocessing_agent)
    workflow.add_node("evaluator", evaluator)
    
    # 시작 노드 설정
    workflow.set_entry_point("data_loader")
    
    # 엣지 설정
    # DataLoader에서 TextAnalysisAgent로
    workflow.add_edge("data_loader", "text_analysis_agent")
    
    # TextAnalysisAgent에서 VisualizationAgent로
    workflow.add_edge("text_analysis_agent", "visualization_agent")
    
    # VisualizationAgent에서 DomainAnalysisAgent로
    workflow.add_edge("visualization_agent", "domain_analysis_agent")
    
    # DomainAnalysisAgent에서 PreprocessingAgent로
    workflow.add_edge("domain_analysis_agent", "preprocessing_agent")
    
    # PreprocessingAgent에서 Evaluator로
    workflow.add_edge("preprocessing_agent", "evaluator")
    
    # Evaluator에서 종료
    workflow.add_edge("evaluator", END)
    
    # 그래프 컴파일
    app = workflow.compile()
    
    return app


def run_workflow(file_path: str) -> Dict[str, Any]:
    """
    전체 워크플로우를 실행합니다.
    
    Args:
        file_path: CSV 파일 경로
        
    Returns:
        실행 결과
    """
    print("="*80)
    print("멀티모달 LLM 에이전트 워크플로우 시작")
    print("="*80)
    
    # 그래프 생성
    app = create_agent_graph()
    
    # 초기 상태 설정
    initial_state = {
        "file_path": file_path,
        "thread_id": "main",
        "checkpoint_ns": "viz2prep",
        "checkpoint_id": "workflow"
    }
    
    # 워크플로우 실행
    print(f"데이터 파일: {file_path}")
    print("\n워크플로우 실행 중...")
    
    try:
        # 그래프 실행
        result = app.invoke(initial_state)
        
        print("\n" + "="*80)
        print("워크플로우 실행 완료!")
        print("="*80)
        
        return result
        
    except Exception as e:
        print(f"워크플로우 실행 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    # 테스트 실행
    import os
    
    # Titanic 데이터셋 다운로드 (없는 경우)
    titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    
    if not os.path.exists("titanic.csv"):
        print("Titanic 데이터셋을 다운로드 중...")
        import urllib.request
        urllib.request.urlretrieve(titanic_url, "titanic.csv")
        print("다운로드 완료!")
    
    # 워크플로우 실행
    result = run_workflow("titanic.csv")
    
    # 결과 출력
    print("\n최종 결과:")
    print(f"- 원본 데이터 크기: {result['raw_dataframe'].shape}")
    print(f"- 전처리된 데이터 크기: {result['preprocessed_dataframe'].shape}")
    print(f"- 생성된 플롯 수: {len(result.get('plot_paths', []))}")
    print(f"- 전처리 코드 길이: {len(result.get('preprocessing_code', ''))}") 