"""
시각화 에이전트
DataFrame의 EDA 플롯을 생성합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any, List
from langchain_core.runnables import RunnableLambda

# matplotlib 백엔드 설정 (GUI 없이 실행)
import matplotlib
matplotlib.use('Agg')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def create_visualizations(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    DataFrame의 다양한 시각화를 생성합니다.
    
    Args:
        inputs: DataFrame이 포함된 입력 딕셔너리
        
    Returns:
        생성된 플롯 파일 경로들이 포함된 딕셔너리
    """
    df = inputs["dataframe"]
    
    # 플롯 저장 디렉토리 생성
    plots_dir = "generated_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_paths = []
    
    # 1. 결측값 히트맵
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
    plt.title('결측값 히트맵')
    plt.tight_layout()
    missing_heatmap_path = os.path.join(plots_dir, "missing_heatmap.png")
    plt.savefig(missing_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths.append(missing_heatmap_path)
    
    # 2. 결측값 바 플롯
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(missing_counts)), missing_percentages)
    plt.xlabel('컬럼')
    plt.ylabel('결측값 비율 (%)')
    plt.title('컬럼별 결측값 비율')
    plt.xticks(range(len(missing_counts)), df.columns, rotation=45, ha='right')
    
    # 값이 있는 바에만 텍스트 표시
    for i, (bar, count) in enumerate(zip(bars, missing_counts)):
        if count > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    missing_bar_path = os.path.join(plots_dir, "missing_bar.png")
    plt.savefig(missing_bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths.append(missing_bar_path)
    
    # 3. 수치형 컬럼 히스토그램
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        for i, col in enumerate(numeric_cols):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'{col} 히스토그램')
            plt.xlabel(col)
            plt.ylabel('빈도')
        
        plt.tight_layout()
        histograms_path = os.path.join(plots_dir, "histograms.png")
        plt.savefig(histograms_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(histograms_path)
    
    # 4. 수치형 컬럼 박스플롯
    if len(numeric_cols) > 0:
        plt.figure(figsize=(15, 6))
        df[numeric_cols].boxplot()
        plt.title('수치형 컬럼 박스플롯')
        plt.xticks(rotation=45)
        plt.tight_layout()
        boxplots_path = os.path.join(plots_dir, "boxplots.png")
        plt.savefig(boxplots_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(boxplots_path)
    
    # 5. 범주형 컬럼 카운트플롯
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        n_cols = min(2, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        for i, col in enumerate(categorical_cols):
            plt.subplot(n_rows, n_cols, i + 1)
            value_counts = df[col].value_counts().head(10)
            plt.bar(range(len(value_counts)), value_counts.values)
            plt.title(f'{col} 상위 10개 값')
            plt.xlabel(col)
            plt.ylabel('빈도')
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
        
        plt.tight_layout()
        countplots_path = os.path.join(plots_dir, "countplots.png")
        plt.savefig(countplots_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(countplots_path)
    
    # 6. 상관관계 히트맵 (수치형 컬럼이 2개 이상인 경우)
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('상관관계 히트맵')
        plt.tight_layout()
        correlation_path = os.path.join(plots_dir, "correlation_heatmap.png")
        plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(correlation_path)
    
    print(f"시각화 생성 완료: {len(plot_paths)}개 플롯")
    
    return {
        **inputs,  # 기존 입력값(예: text_analysis 등) 유지
        "dataframe": df,
        "plot_paths": plot_paths
    }


# LangGraph 노드로 사용할 수 있는 함수
visualization_agent = RunnableLambda(create_visualizations) 