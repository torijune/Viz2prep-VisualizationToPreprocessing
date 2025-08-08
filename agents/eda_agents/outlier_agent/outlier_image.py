"""
이상치 데이터 분석 이미지 에이전트
박스플롯, Z-Score 분포, 이상치 히트맵 등을 생성하여 시각적 분석 결과를 제공합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from langchain_core.runnables import RunnableLambda
import os


def create_outlier_visualizations(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    이상치의 시각적 분석을 수행하고 이미지를 생성합니다.
    
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
            "outlier_images": []
        }
    
    # 이미지 저장 디렉토리 생성
    os.makedirs("generated_plots", exist_ok=True)
    
    image_paths = []
    
    # 1. 박스플롯 (IQR 기반 이상치 시각화)
    plt.figure(figsize=(15, 10))
    n_cols = min(3, len(numeric_columns))
    n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
    
    for i, col in enumerate(numeric_columns):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # 박스플롯 생성
        box_plot = plt.boxplot(df[col].dropna(), patch_artist=True)
        
        # 박스 색상 설정
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['medians'][0].set_color('red')
        box_plot['fliers'][0].set_markerfacecolor('red')
        box_plot['fliers'][0].set_markeredgecolor('red')
        
        plt.title(f'{col} Boxplot')
        plt.ylabel(col)
        
        # 이상치 개수 표시
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        if len(outliers) > 0:
            plt.text(0.02, 0.98, f'Outliers: {len(outliers)}', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    boxplot_path = "generated_plots/outlier_boxplots.png"
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(boxplot_path)
    
    # 2. Z-Score 분포 히스토그램
    plt.figure(figsize=(15, 10))
    n_cols = min(3, len(numeric_columns))
    n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
    
    for i, col in enumerate(numeric_columns):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Z-Score 계산
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        
        # 히스토그램 그리기
        plt.hist(z_scores, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(x=-3, color='red', linestyle='--', alpha=0.7, label='Z=-3')
        plt.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='Z=3')
        plt.axvline(x=0, color='green', linestyle='-', alpha=0.7, label='Mean')
        
        plt.title(f'{col} Z-Score Distribution')
        plt.xlabel('Z-Score')
        plt.ylabel('Frequency')
        plt.legend()
        
        # 이상치 개수 표시
        outliers = df[np.abs(z_scores) > 3]
        if len(outliers) > 0:
            plt.text(0.02, 0.98, f'Outliers: {len(outliers)}', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    zscore_path = "generated_plots/outlier_zscores.png"
    plt.savefig(zscore_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(zscore_path)
    
    # 3. 이상치 비교 분석
    plt.figure(figsize=(15, 10))
    
    # IQR과 Z-Score 결과 비교
    comparison_data = []
    for col in numeric_columns:
        # IQR 이상치
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        # Z-Score 이상치
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        zscore_outliers = df[z_scores > 3]
        
        comparison_data.append({
            'column': col,
            'iqr_outliers': len(iqr_outliers),
            'zscore_outliers': len(zscore_outliers),
            'total_data': len(df)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 서브플롯 생성
    plt.subplot(2, 2, 1)
    x = np.arange(len(comparison_df))
    width = 0.35
    plt.bar(x - width/2, comparison_df['iqr_outliers'], width, label='IQR', alpha=0.7)
    plt.bar(x + width/2, comparison_df['zscore_outliers'], width, label='Z-Score', alpha=0.7)
    plt.title('IQR vs Z-Score Outlier Count Comparison')
    plt.xlabel('Variable')
    plt.ylabel('Outlier Count')
    plt.xticks(x, comparison_df['column'], rotation=45, ha='right')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    iqr_percentages = (comparison_df['iqr_outliers'] / comparison_df['total_data']) * 100
    zscore_percentages = (comparison_df['zscore_outliers'] / comparison_df['total_data']) * 100
    plt.bar(x - width/2, iqr_percentages, width, label='IQR', alpha=0.7)
    plt.bar(x + width/2, zscore_percentages, width, label='Z-Score', alpha=0.7)
    plt.title('IQR vs Z-Score Outlier Ratio Comparison')
    plt.xlabel('Variable')
    plt.ylabel('Outlier Ratio (%)')
    plt.xticks(x, comparison_df['column'], rotation=45, ha='right')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    difference = comparison_df['iqr_outliers'] - comparison_df['zscore_outliers']
    colors = ['red' if x > 0 else 'blue' for x in difference]
    plt.bar(x, difference, color=colors, alpha=0.7)
    plt.title('IQR - Z-Score Outlier Count Difference')
    plt.xlabel('Variable')
    plt.ylabel('Count Difference')
    plt.xticks(x, comparison_df['column'], rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.subplot(2, 2, 4)
    total_outliers = comparison_df['iqr_outliers'] + comparison_df['zscore_outliers']
    plt.pie(total_outliers, labels=comparison_df['column'], autopct='%1.1f%%')
    plt.title('Total Outlier Distribution')
    
    plt.tight_layout()
    comparison_path = "generated_plots/outlier_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(comparison_path)
    
    # 4. 이상치 히트맵
    plt.figure(figsize=(12, 8))
    
    # 이상치 매트릭스 생성
    outlier_matrix = pd.DataFrame(index=df.index, columns=numeric_columns)
    
    for col in numeric_columns:
        # IQR 이상치
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        # Z-Score 이상치
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        zscore_outliers = z_scores > 3
        
        # 이상치 매트릭스 (0: 정상, 1: IQR 이상치, 2: Z-Score 이상치, 3: 둘 다)
        outlier_matrix[col] = iqr_outliers.astype(int) + zscore_outliers.astype(int) * 2
    
    # 히트맵 생성
    sns.heatmap(outlier_matrix, cbar=True, cmap='RdYlBu_r', 
               cbar_kws={'label': 'Outlier Type'})
    plt.title('Outlier Heatmap (0: Normal, 1: IQR, 2: Z-Score, 3: Both)')
    plt.xlabel('Variable')
    plt.ylabel('Row')
    plt.tight_layout()
    heatmap_path = "generated_plots/outlier_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(heatmap_path)
    
    # 5. 이상치 처리 전략 시각화
    plt.figure(figsize=(15, 10))
    
    for i, col in enumerate(numeric_columns):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # 원본 데이터 분포
        plt.hist(df[col], bins=30, alpha=0.7, label='Original', color='blue')
        
        # 이상치 제거 후 분포
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        clean_data = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)][col]
        
        if len(clean_data) > 0:
            plt.hist(clean_data, bins=30, alpha=0.7, label='Outlier Removed', color='green')
        
        plt.title(f'{col} Outlier Handling Comparison')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.legend()
        
        # 이상치 비율 표시
        outlier_ratio = (len(df) - len(clean_data)) / len(df) * 100
        plt.text(0.02, 0.98, f'Outliers: {outlier_ratio:.1f}%', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    strategy_path = "generated_plots/outlier_strategy.png"
    plt.savefig(strategy_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(strategy_path)
    
    # 6. 이상치 수준 분류
    plt.figure(figsize=(12, 8))
    
    # 이상치 수준 분류
    outlier_levels = []
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_ratio = len(outliers) / len(df) * 100
        
        if outlier_ratio < 5:
            level = 'Low'
        elif outlier_ratio < 15:
            level = 'Normal'
        else:
            level = 'High'
        
        outlier_levels.append({
            'column': col,
            'outlier_ratio': outlier_ratio,
            'level': level
        })
    
    outlier_df = pd.DataFrame(outlier_levels)
    
    # 수준별 분포
    plt.subplot(2, 2, 1)
    level_counts = outlier_df['level'].value_counts()
    plt.pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%')
    plt.title('Outlier Level Distribution')
    
    # 이상치 비율 분포
    plt.subplot(2, 2, 2)
    plt.bar(outlier_df['column'], outlier_df['outlier_ratio'])
    plt.title('Variable Outlier Ratio')
    plt.ylabel('Outlier Ratio (%)')
    plt.xticks(rotation=45, ha='right')
    
    # 수준별 색상 구분
    plt.subplot(2, 2, 3)
    colors = {'Low': 'green', 'Normal': 'orange', 'High': 'red'}
    color_list = [colors[level] for level in outlier_df['level']]
    plt.bar(outlier_df['column'], outlier_df['outlier_ratio'], color=color_list)
    plt.title('Outlier Level Color Coding')
    plt.ylabel('Outlier Ratio (%)')
    plt.xticks(rotation=45, ha='right')
    
    # 이상치 비율 히스토그램
    plt.subplot(2, 2, 4)
    plt.hist(outlier_df['outlier_ratio'], bins=10, alpha=0.7, edgecolor='black')
    plt.title('Outlier Ratio Distribution')
    plt.xlabel('Outlier Ratio (%)')
    plt.ylabel('Variable Count')
    
    plt.tight_layout()
    level_path = "generated_plots/outlier_levels.png"
    plt.savefig(level_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(level_path)
    
    print(f"Outlier Visualization Complete: {len(image_paths)} images generated")
    
    return {
        **inputs,
        "outlier_images": image_paths
    }


# LangGraph 노드로 사용할 수 있는 함수
outlier_image_agent = RunnableLambda(create_outlier_visualizations)
