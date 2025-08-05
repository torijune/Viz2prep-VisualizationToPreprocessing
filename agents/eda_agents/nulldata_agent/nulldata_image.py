"""
결측치 및 중복 데이터 분석 이미지 에이전트
결측치 히트맵, 중복 데이터 시각화 등을 생성하여 시각적 분석 결과를 제공합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from langchain_core.runnables import RunnableLambda
import os


def create_missing_duplicate_visualizations(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    결측치와 중복 데이터의 시각적 분석을 수행하고 이미지를 생성합니다.
    
    Args:
        inputs: DataFrame이 포함된 입력 딕셔너리
        
    Returns:
        생성된 이미지 경로들이 포함된 딕셔너리
    """
    df = inputs["dataframe"]
    
    # 이미지 저장 디렉토리 생성
    os.makedirs("generated_plots", exist_ok=True)
    
    image_paths = []
    
    # 1. 결측치 히트맵
    plt.figure(figsize=(12, 8))
    missing_matrix = df.isnull()
    sns.heatmap(missing_matrix, cbar=True, yticklabels=False, cmap='viridis')
    plt.title('결측치 히트맵')
    plt.xlabel('컬럼')
    plt.ylabel('행')
    plt.tight_layout()
    missing_heatmap_path = "generated_plots/missing_heatmap.png"
    plt.savefig(missing_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(missing_heatmap_path)
    
    # 2. 컬럼별 결측치 막대그래프
    plt.figure(figsize=(12, 6))
    missing_by_column = df.isnull().sum()
    missing_percentage_by_column = (missing_by_column / len(df)) * 100
    
    # 결측치가 있는 컬럼만 선택
    columns_with_missing = missing_by_column[missing_by_column > 0].index
    
    if len(columns_with_missing) > 0:
        plt.subplot(1, 2, 1)
        plt.bar(columns_with_missing, missing_by_column[columns_with_missing])
        plt.title('컬럼별 결측치 개수')
        plt.ylabel('결측치 개수')
        plt.xticks(rotation=45, ha='right')
        
        plt.subplot(1, 2, 2)
        plt.bar(columns_with_missing, missing_percentage_by_column[columns_with_missing])
        plt.title('컬럼별 결측치 비율')
        plt.ylabel('결측치 비율 (%)')
        plt.xticks(rotation=45, ha='right')
    else:
        plt.text(0.5, 0.5, '결측치가 없습니다', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('결측치 분석')
    
    plt.tight_layout()
    missing_bar_path = "generated_plots/missing_bar.png"
    plt.savefig(missing_bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(missing_bar_path)
    
    # 3. 결측치 패턴 분석
    plt.figure(figsize=(12, 8))
    
    # 결측치 개수별 행 분포
    missing_counts = df.isnull().sum(axis=1)
    missing_count_distribution = missing_counts.value_counts().sort_index()
    
    plt.subplot(2, 2, 1)
    plt.bar(missing_count_distribution.index, missing_count_distribution.values)
    plt.title('결측치 개수별 행 분포')
    plt.xlabel('결측치 개수')
    plt.ylabel('행 개수')
    
    # 결측치 비율 분포
    plt.subplot(2, 2, 2)
    missing_ratio = (missing_counts / len(df.columns)) * 100
    plt.hist(missing_ratio, bins=20, alpha=0.7, edgecolor='black')
    plt.title('행별 결측치 비율 분포')
    plt.xlabel('결측치 비율 (%)')
    plt.ylabel('행 개수')
    
    # 완전한 행 vs 불완전한 행
    plt.subplot(2, 2, 3)
    complete_rows = (missing_counts == 0).sum()
    incomplete_rows = len(df) - complete_rows
    plt.pie([complete_rows, incomplete_rows], labels=['완전한 행', '불완전한 행'], autopct='%1.1f%%')
    plt.title('행 완전성 분포')
    
    # 결측치가 있는 컬럼 수
    plt.subplot(2, 2, 4)
    columns_with_missing_count = (missing_by_column > 0).sum()
    columns_without_missing_count = len(df.columns) - columns_with_missing_count
    plt.pie([columns_without_missing_count, columns_with_missing_count], 
            labels=['결측치 없는 컬럼', '결측치 있는 컬럼'], autopct='%1.1f%%')
    plt.title('컬럼 결측치 분포')
    
    plt.tight_layout()
    missing_pattern_path = "generated_plots/missing_pattern.png"
    plt.savefig(missing_pattern_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(missing_pattern_path)
    
    # 4. 중복 데이터 분석
    plt.figure(figsize=(12, 8))
    
    # 중복 행 개수
    total_duplicates = df.duplicated().sum()
    unique_rows = len(df) - total_duplicates
    
    plt.subplot(2, 2, 1)
    plt.pie([unique_rows, total_duplicates], labels=['고유 행', '중복 행'], autopct='%1.1f%%')
    plt.title('중복 데이터 분포')
    
    # 중복 패턴 분석
    if total_duplicates > 0:
        # 중복 그룹 분석
        duplicate_groups = df[df.duplicated(keep=False)].groupby(df.columns.tolist()).size()
        
        plt.subplot(2, 2, 2)
        plt.hist(duplicate_groups.values, bins=min(20, len(duplicate_groups)), alpha=0.7, edgecolor='black')
        plt.title('중복 그룹 크기 분포')
        plt.xlabel('중복 그룹 크기')
        plt.ylabel('그룹 개수')
        
        # 중복 패턴 히트맵 (상위 10개)
        plt.subplot(2, 2, 3)
        top_duplicates = duplicate_groups.head(10)
        plt.bar(range(len(top_duplicates)), top_duplicates.values)
        plt.title('상위 10개 중복 패턴')
        plt.ylabel('중복 횟수')
        plt.xticks(range(len(top_duplicates)), [f'패턴 {i+1}' for i in range(len(top_duplicates))], rotation=45)
    else:
        plt.subplot(2, 2, 2)
        plt.text(0.5, 0.5, '중복 데이터 없음', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('중복 그룹 분석')
        
        plt.subplot(2, 2, 3)
        plt.text(0.5, 0.5, '중복 데이터 없음', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('중복 패턴 분석')
    
    # 데이터 품질 점수
    plt.subplot(2, 2, 4)
    missing_quality_score = 100 - (df.isnull().sum().sum() / df.size) * 100
    duplicate_quality_score = 100 - (df.duplicated().sum() / len(df)) * 100
    
    quality_scores = [missing_quality_score, duplicate_quality_score]
    quality_labels = ['결측치 품질', '중복 데이터 품질']
    colors = ['lightblue', 'lightcoral']
    
    plt.bar(quality_labels, quality_scores, color=colors)
    plt.title('데이터 품질 점수')
    plt.ylabel('품질 점수 (%)')
    plt.ylim(0, 100)
    
    # 점수 표시
    for i, score in enumerate(quality_scores):
        plt.text(i, score + 1, f'{score:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    duplicate_analysis_path = "generated_plots/duplicate_analysis.png"
    plt.savefig(duplicate_analysis_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(duplicate_analysis_path)
    
    # 5. 결측치 상관관계 분석
    if len(columns_with_missing) >= 2:
        plt.figure(figsize=(10, 8))
        
        # 결측치 간 상관관계 계산
        missing_correlation = df[columns_with_missing].isnull().corr()
        
        sns.heatmap(missing_correlation, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('결측치 간 상관관계')
        plt.tight_layout()
        missing_corr_path = "generated_plots/missing_correlation.png"
        plt.savefig(missing_corr_path, dpi=300, bbox_inches='tight')
        plt.close()
        image_paths.append(missing_corr_path)
    
    print(f"결측치 및 중복 데이터 시각화 완료: {len(image_paths)}개 이미지 생성")
    
    return {
        **inputs,
        "missing_duplicate_images": image_paths
    }


# LangGraph 노드로 사용할 수 있는 함수
missing_duplicate_image_agent = RunnableLambda(create_missing_duplicate_visualizations) 