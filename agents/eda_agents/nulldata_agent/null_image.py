"""
결측값 시각화 이미지 에이전트
결측값 분포를 시각적으로 분석하는 이미지 생성 에이전트
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any, List
from langchain_core.runnables import RunnableLambda

# matplotlib 백엔드 설정
import matplotlib
matplotlib.use('Agg')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def create_null_visualizations(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    결측값 분포를 시각화하는 함수
    
    Args:
        inputs: DataFrame이 포함된 입력 딕셔너리
        
    Returns:
        생성된 이미지 경로들이 포함된 딕셔너리
    """
    df = inputs["dataframe"]
    
    # 이미지 저장 디렉토리 생성
    os.makedirs("generated_plots", exist_ok=True)
    
    image_paths = []
    
    # 1. 결측값 히트맵
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
    plt.title('결측값 히트맵')
    plt.tight_layout()
    null_heatmap_path = os.path.join("generated_plots", "null_heatmap.png")
    plt.savefig(null_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(null_heatmap_path)
    
    # 2. 컬럼별 결측값 비율 바 플롯
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
    null_bar_path = os.path.join("generated_plots", "null_bar.png")
    plt.savefig(null_bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(null_bar_path)
    
    # 3. 결측값 패턴 분석 (상관관계)
    if df.isnull().sum().sum() > 0:
        # 결측값이 있는 컬럼들만 선택
        null_cols = df.columns[df.isnull().sum() > 0].tolist()
        if len(null_cols) > 1:
            # 결측값 간 상관관계 계산
            null_corr = df[null_cols].isnull().corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(null_corr, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5)
            plt.title('결측값 패턴 상관관계')
            plt.tight_layout()
            null_pattern_path = os.path.join("generated_plots", "null_pattern.png")
            plt.savefig(null_pattern_path, dpi=300, bbox_inches='tight')
            plt.close()
            image_paths.append(null_pattern_path)
    
    # 4. 결측값 분포 파이 차트
    total_missing = df.isnull().sum().sum()
    total_cells = len(df) * len(df.columns)
    missing_ratio = (total_missing / total_cells) * 100
    
    plt.figure(figsize=(8, 8))
    labels = ['결측값', '유효값']
    sizes = [missing_ratio, 100 - missing_ratio]
    colors = ['#ff9999', '#66b3ff']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('전체 데이터 결측값 비율')
    plt.axis('equal')
    
    null_pie_path = os.path.join("generated_plots", "null_pie.png")
    plt.savefig(null_pie_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(null_pie_path)
    
    print(f"결측값 시각화 생성 완료: {len(image_paths)}개 이미지")
    
    return {
        **inputs,
        "dataframe": df,
        "null_image_paths": image_paths
    }


# LangGraph 노드로 사용할 수 있는 함수
null_image_agent = RunnableLambda(create_null_visualizations)
