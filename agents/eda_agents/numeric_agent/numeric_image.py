"""
연속형 데이터 통계치 분석 이미지 에이전트
히스토그램, 박스플롯, 분포도 등을 생성하여 시각적 분석 결과를 제공합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from langchain_core.runnables import RunnableLambda
import os


def create_numeric_visualizations(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    연속형 데이터의 시각적 분석을 수행하고 이미지를 생성합니다.
    
    Args:
        inputs: DataFrame이 포함된 입력 딕셔너리
        
    Returns:
        생성된 이미지 경로들이 포함된 딕셔너리
    """
    df = inputs["dataframe"]
    
    # 수치형 컬럼만 선택
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_columns:
        return {
            **inputs,
            "numeric_images": []
        }
    
    # 이미지 저장 디렉토리 생성
    os.makedirs("generated_plots", exist_ok=True)
    
    image_paths = []
    
    # 1. 히스토그램 (분포 확인)
    plt.figure(figsize=(15, 10))
    n_cols = min(3, len(numeric_columns))
    n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
    
    for i, col in enumerate(numeric_columns):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
        plt.title(f'{col} 분포')
        plt.xlabel(col)
        plt.ylabel('빈도')
        
        # 통계 정보 추가
        mean_val = df[col].mean()
        std_val = df[col].std()
        plt.axvline(mean_val, color='red', linestyle='--', label=f'평균: {mean_val:.2f}')
        plt.legend()
    
    plt.tight_layout()
    histogram_path = "generated_plots/numeric_histograms.png"
    plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(histogram_path)
    
    # 2. 박스플롯 (이상치 확인)
    plt.figure(figsize=(15, 8))
    n_cols = min(3, len(numeric_columns))
    n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
    
    for i, col in enumerate(numeric_columns):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.boxplot(df[col].dropna())
        plt.title(f'{col} 박스플롯')
        plt.ylabel(col)
    
    plt.tight_layout()
    boxplot_path = "generated_plots/numeric_boxplots.png"
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(boxplot_path)
    
    # 3. Q-Q 플롯 (정규성 검정)
    if len(numeric_columns) > 0:
        plt.figure(figsize=(15, 10))
        n_cols = min(3, len(numeric_columns))
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
        
        for i, col in enumerate(numeric_columns):
            plt.subplot(n_rows, n_cols, i + 1)
            from scipy import stats
            stats.probplot(df[col].dropna(), dist="norm", plot=plt)
            plt.title(f'{col} Q-Q Plot')
        
        plt.tight_layout()
        qqplot_path = "generated_plots/numeric_qqplots.png"
        plt.savefig(qqplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        image_paths.append(qqplot_path)
    
    # 4. 상관관계 히트맵 (수치형 컬럼이 2개 이상인 경우)
    if len(numeric_columns) >= 2:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numeric_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('수치형 변수 상관관계 히트맵')
        plt.tight_layout()
        corr_path = "generated_plots/numeric_correlation.png"
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        plt.close()
        image_paths.append(corr_path)
    
    # 5. 기술통계 요약 시각화
    plt.figure(figsize=(15, 10))
    
    # 통계 데이터 준비
    stats_data = []
    for col in numeric_columns:
        stats = df[col].describe()
        stats_data.append({
            'column': col,
            'mean': stats['mean'],
            'std': stats['std'],
            'min': stats['min'],
            'max': stats['max'],
            'skewness': df[col].skew(),
            'kurtosis': df[col].kurtosis()
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    # 서브플롯 생성
    plt.subplot(2, 2, 1)
    plt.bar(stats_df['column'], stats_df['mean'])
    plt.title('평균값 비교')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    plt.bar(stats_df['column'], stats_df['std'])
    plt.title('표준편차 비교')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 3)
    plt.bar(stats_df['column'], stats_df['skewness'])
    plt.title('왜도 비교')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 4)
    plt.bar(stats_df['column'], stats_df['kurtosis'])
    plt.title('첨도 비교')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    stats_summary_path = "generated_plots/numeric_stats_summary.png"
    plt.savefig(stats_summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(stats_summary_path)
    
    print(f"연속형 데이터 시각화 완료: {len(image_paths)}개 이미지 생성")
    
    return {
        **inputs,
        "numeric_images": image_paths
    }


# LangGraph 노드로 사용할 수 있는 함수
numeric_image_agent = RunnableLambda(create_numeric_visualizations)
