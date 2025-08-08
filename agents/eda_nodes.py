#!/usr/bin/env python3
"""
EDA 노드들 - LangGraph용 EDA 에이전트들
기존 EDA 에이전트들을 LangGraph 노드로 변환하여 structured format으로 결과를 반환
"""

import os
import sys
from typing import Dict, Any
import pandas as pd

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 기존 EDA 에이전트들 import
from agents.eda_agents.numeric_agent.numeric_text import analyze_numeric_statistics
from agents.eda_agents.numeric_agent.numeric_image import create_numeric_visualizations
from agents.eda_agents.category_agent.cate_text import analyze_categorical_data
from agents.eda_agents.category_agent.cate_image import create_categorical_visualizations
from agents.eda_agents.nulldata_agent.null_text import analyze_null_data_text
from agents.eda_agents.nulldata_agent.null_image import create_null_visualizations
from agents.eda_agents.corr_agent.corr_text import analyze_correlations
from agents.eda_agents.corr_agent.corr_image import create_correlation_visualizations
from agents.eda_agents.outlier_agent.outlier_text import analyze_outliers
from agents.eda_agents.outlier_agent.outlier_image import create_outlier_visualizations

from workflow_state import WorkflowState

def numeric_agent_node(state: WorkflowState) -> WorkflowState:
    """
    수치형 변수 EDA 노드
    
    🔍 INPUT STATES:
    - dataframe: 원본 데이터프레임
    
    📊 OUTPUT STATES:
    - numeric_eda_result: 수치형 변수 EDA 결과
    
    ➡️ NEXT EDGE: numeric_planner
    """
    print("🔢 [EDA] 수치형 변수 분석 시작...")
    
    try:
        dataframe = state["dataframe"]
        
        # 텍스트 분석 수행
        text_result = analyze_numeric_statistics({"dataframe": dataframe})
        
        # 이미지 분석 수행
        image_result = create_numeric_visualizations({"dataframe": dataframe})
        
        # structured format으로 결과 정리
        numeric_eda_result = {
            "text_analysis": text_result.get("text_analysis", {}),
            "visualizations": image_result.get("image_paths", []),
            "statistics": text_result.get("statistics", {}),
            "insights": text_result.get("insights", []),
            "columns_analyzed": text_result.get("columns_analyzed", []),
            "status": "success"
        }
        
        print(f"✅ [EDA] 수치형 변수 분석 완료 - {len(numeric_eda_result.get('columns_analyzed', []))}개 변수 분석")
        
        return {
            **state,
            "numeric_eda_result": numeric_eda_result
        }
        
    except Exception as e:
        print(f"❌ [EDA] 수치형 변수 분석 오류: {e}")
        return {
            **state,
            "numeric_eda_result": {
                "status": "error",
                "error_message": str(e),
                "text_analysis": {},
                "visualizations": [],
                "statistics": {},
                "insights": [],
                "columns_analyzed": []
            }
        }


def category_agent_node(state: WorkflowState) -> WorkflowState:
    """
    범주형 변수 EDA 노드
    
    🔍 INPUT STATES:
    - dataframe: 원본 데이터프레임
    
    📊 OUTPUT STATES:
    - category_eda_result: 범주형 변수 EDA 결과
    
    ➡️ NEXT EDGE: category_planner
    """
    print("📊 [EDA] 범주형 변수 분석 시작...")
    
    try:
        dataframe = state["dataframe"]
        
        # 텍스트 분석 수행
        text_result = analyze_categorical_data({"dataframe": dataframe})
        
        # 이미지 분석 수행
        image_result = create_categorical_visualizations({"dataframe": dataframe})
        
        # structured format으로 결과 정리
        category_eda_result = {
            "text_analysis": text_result.get("text_analysis", {}),
            "visualizations": image_result.get("image_paths", []),
            "categorical_summary": text_result.get("categorical_summary", {}),
            "insights": text_result.get("insights", []),
            "columns_analyzed": text_result.get("columns_analyzed", []),
            "status": "success"
        }
        
        print(f"✅ [EDA] 범주형 변수 분석 완료 - {len(category_eda_result.get('columns_analyzed', []))}개 변수 분석")
        
        return {
            **state,
            "category_eda_result": category_eda_result
        }
        
    except Exception as e:
        print(f"❌ [EDA] 범주형 변수 분석 오류: {e}")
        return {
            **state,
            "category_eda_result": {
                "status": "error",
                "error_message": str(e),
                "text_analysis": {},
                "visualizations": [],
                "categorical_summary": {},
                "insights": [],
                "columns_analyzed": []
            }
        }


def outlier_agent_node(state: WorkflowState) -> WorkflowState:
    """
    이상치 EDA 노드
    
    🔍 INPUT STATES:
    - dataframe: 원본 데이터프레임
    
    📊 OUTPUT STATES:
    - outlier_eda_result: 이상치 EDA 결과
    
    ➡️ NEXT EDGE: outlier_planner
    """
    print("🚨 [EDA] 이상치 분석 시작...")
    
    try:
        dataframe = state["dataframe"]
        
        # 텍스트 분석 수행
        text_result = analyze_outliers({"dataframe": dataframe})
        
        # 이미지 분석 수행
        image_result = create_outlier_visualizations({"dataframe": dataframe})
        
        # structured format으로 결과 정리
        outlier_eda_result = {
            "text_analysis": text_result.get("text_analysis", {}),
            "visualizations": image_result.get("image_paths", []),
            "outlier_summary": text_result.get("outlier_summary", {}),
            "insights": text_result.get("insights", []),
            "columns_analyzed": text_result.get("columns_analyzed", []),
            "total_outliers": text_result.get("total_outliers", 0),
            "status": "success"
        }
        
        print(f"✅ [EDA] 이상치 분석 완료 - {outlier_eda_result.get('total_outliers', 0)}개 이상치 발견")
        
        return {
            **state,
            "outlier_eda_result": outlier_eda_result
        }
        
    except Exception as e:
        print(f"❌ [EDA] 이상치 분석 오류: {e}")
        return {
            **state,
            "outlier_eda_result": {
                "status": "error",
                "error_message": str(e),
                "text_analysis": {},
                "visualizations": [],
                "outlier_summary": {},
                "insights": [],
                "columns_analyzed": [],
                "total_outliers": 0
            }
        }


def nulldata_agent_node(state: WorkflowState) -> WorkflowState:
    """
    결측값 EDA 노드
    
    🔍 INPUT STATES:
    - dataframe: 원본 데이터프레임
    
    📊 OUTPUT STATES:
    - nulldata_eda_result: 결측값 EDA 결과
    
    ➡️ NEXT EDGE: nulldata_planner
    """
    print("❓ [EDA] 결측값 분석 시작...")
    
    try:
        dataframe = state["dataframe"]
        
        # 텍스트 분석 수행
        text_result = analyze_null_data_text({"dataframe": dataframe})
        
        # 이미지 분석 수행
        image_result = create_null_visualizations({"dataframe": dataframe})
        
        # structured format으로 결과 정리
        nulldata_eda_result = {
            "text_analysis": text_result.get("text_analysis", {}),
            "visualizations": image_result.get("image_paths", []),
            "null_summary": text_result.get("null_summary", {}),
            "insights": text_result.get("insights", []),
            "columns_with_nulls": text_result.get("columns_with_nulls", []),
            "total_missing": text_result.get("total_missing", 0),
            "status": "success"
        }
        
        print(f"✅ [EDA] 결측값 분석 완료 - {nulldata_eda_result.get('total_missing', 0)}개 결측값 발견")
        
        return {
            **state,
            "nulldata_eda_result": nulldata_eda_result
        }
        
    except Exception as e:
        print(f"❌ [EDA] 결측값 분석 오류: {e}")
        return {
            **state,
            "nulldata_eda_result": {
                "status": "error",
                "error_message": str(e),
                "text_analysis": {},
                "visualizations": [],
                "null_summary": {},
                "insights": [],
                "columns_with_nulls": [],
                "total_missing": 0
            }
        }


def corr_agent_node(state: WorkflowState) -> WorkflowState:
    """
    상관관계 EDA 노드
    
    🔍 INPUT STATES:
    - dataframe: 원본 데이터프레임
    
    📊 OUTPUT STATES:
    - corr_eda_result: 상관관계 EDA 결과
    
    ➡️ NEXT EDGE: corr_planner
    """
    print("🔗 [EDA] 상관관계 분석 시작...")
    
    try:
        dataframe = state["dataframe"]
        
        # 텍스트 분석 수행
        text_result = analyze_correlations({"dataframe": dataframe})
        
        # 이미지 분석 수행
        image_result = create_correlation_visualizations({"dataframe": dataframe})
        
        # structured format으로 결과 정리
        corr_eda_result = {
            "text_analysis": text_result.get("text_analysis", {}),
            "visualizations": image_result.get("image_paths", []),
            "correlation_matrix": text_result.get("correlation_matrix", {}),
            "insights": text_result.get("insights", []),
            "high_correlations": text_result.get("high_correlations", []),
            "status": "success"
        }
        
        print(f"✅ [EDA] 상관관계 분석 완료 - {len(corr_eda_result.get('high_correlations', []))}개 강한 상관관계 발견")
        
        return {
            **state,
            "corr_eda_result": corr_eda_result
        }
        
    except Exception as e:
        print(f"❌ [EDA] 상관관계 분석 오류: {e}")
        return {
            **state,
            "corr_eda_result": {
                "status": "error",
                "error_message": str(e),
                "text_analysis": {},
                "visualizations": [],
                "correlation_matrix": {},
                "insights": [],
                "high_correlations": []
            }
        } 