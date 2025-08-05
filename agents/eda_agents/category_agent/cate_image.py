"""
범주형 데이터 분석 이미지 에이전트
막대그래프, 파이차트, 카테고리 분포도 등을 생성하여 시각적 분석 결과를 제공합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from langchain_core.runnables import RunnableLambda
import os


def create_categorical_visualizations(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    범주형 데이터의 시각적 분석을 수행하고 이미지를 생성합니다.
    
    Args:
        inputs: DataFrame이 포함된 입력 딕셔너리
        
    Returns:
        생성된 이미지 경로들이 포함된 딕셔너리
    """
    df = inputs["dataframe"]
    
    # 범주형 컬럼 선택 (object, category 타입)
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_columns:
        return {
            **inputs,
            "categorical_images": []
        }
    
    # 이미지 저장 디렉토리 생성
    os.makedirs("generated_plots", exist_ok=True)
    
    image_paths = []
    
    # 1. 범주형 변수별 분포 막대그래프
    plt.figure(figsize=(15, 10))
    n_cols = min(3, len(categorical_columns))
    n_rows = (len(categorical_columns) + n_cols - 1) // n_cols
    
    for i, col in enumerate(categorical_columns):
        plt.subplot(n_rows, n_cols, i + 1)
        value_counts = df[col].value_counts()
        
        # 상위 10개만 표시 (너무 많은 경우)
        if len(value_counts) > 10:
            top_values = value_counts.head(10)
            plt.bar(range(len(top_values)), top_values.values)
            plt.xticks(range(len(top_values)), top_values.index, rotation=45, ha='right')
        else:
            plt.bar(range(len(value_counts)), value_counts.values)
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
        
        plt.title(f'{col} 분포')
        plt.ylabel('개수')
        plt.tight_layout()
    
    barplot_path = "generated_plots/categorical_barplots.png"
    plt.savefig(barplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(barplot_path)
    
    # 2. 파이차트 (상위 5개 카테고리)
    plt.figure(figsize=(15, 10))
    n_cols = min(3, len(categorical_columns))
    n_rows = (len(categorical_columns) + n_cols - 1) // n_cols
    
    for i, col in enumerate(categorical_columns):
        plt.subplot(n_rows, n_cols, i + 1)
        value_counts = df[col].value_counts()
        
        # 상위 5개만 파이차트로 표시
        top_5 = value_counts.head(5)
        if len(value_counts) > 5:
            others = value_counts.iloc[5:].sum()
            top_5_with_others = pd.concat([top_5, pd.Series({'기타': others})])
        else:
            top_5_with_others = top_5
        
        plt.pie(top_5_with_others.values, labels=top_5_with_others.index, autopct='%1.1f%%')
        plt.title(f'{col} 상위 5개 분포')
    
    plt.tight_layout()
    piechart_path = "generated_plots/categorical_piecharts.png"
    plt.savefig(piechart_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(piechart_path)
    
    # 3. 카테고리 불균형 히트맵
    if len(categorical_columns) >= 2:
        plt.figure(figsize=(12, 8))
        
        # 카테고리 불균형 지표 계산
        imbalance_data = []
        for col in categorical_columns:
            value_counts = df[col].value_counts()
            total = len(df[col])
            proportions = value_counts / total
            entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
            max_entropy = np.log2(len(value_counts))
            balance_ratio = entropy / max_entropy if max_entropy > 0 else 0
            imbalance_data.append({
                'column': col,
                'unique_count': len(value_counts),
                'balance_ratio': balance_ratio,
                'entropy': entropy
            })
        
        imbalance_df = pd.DataFrame(imbalance_data)
        
        # 히트맵 생성
        plt.figure(figsize=(10, 6))
        sns.heatmap(imbalance_df.set_index('column')[['balance_ratio', 'entropy']], 
                   annot=True, cmap='RdYlBu_r', center=0.5)
        plt.title('범주형 변수 불균형 지표')
        plt.tight_layout()
        imbalance_path = "generated_plots/categorical_imbalance.png"
        plt.savefig(imbalance_path, dpi=300, bbox_inches='tight')
        plt.close()
        image_paths.append(imbalance_path)
    
    # 4. 범주형 변수 간 관계 히트맵
    if len(categorical_columns) >= 2:
        plt.figure(figsize=(12, 10))
        
        # 카이제곱 통계 계산
        chi_square_matrix = pd.DataFrame(index=categorical_columns, columns=categorical_columns)
        
        for col1 in categorical_columns:
            for col2 in categorical_columns:
                if col1 == col2:
                    chi_square_matrix.loc[col1, col2] = 1.0
                else:
                    # 카이제곱 통계 계산
                    contingency_table = pd.crosstab(df[col1], df[col2])
                    from scipy.stats import chi2_contingency
                    try:
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                        # p-value를 기반으로 관계 강도 계산
                        relationship_strength = 1 - p_value if p_value < 0.05 else 0
                        chi_square_matrix.loc[col1, col2] = relationship_strength
                    except:
                        chi_square_matrix.loc[col1, col2] = 0
        
        sns.heatmap(chi_square_matrix, annot=True, cmap='Blues', vmin=0, vmax=1)
        plt.title('범주형 변수 간 관계 강도 (카이제곱 기반)')
        plt.tight_layout()
        relationship_path = "generated_plots/categorical_relationships.png"
        plt.savefig(relationship_path, dpi=300, bbox_inches='tight')
        plt.close()
        image_paths.append(relationship_path)
    
    # 5. 카디널리티 분석
    plt.figure(figsize=(12, 8))
    
    cardinality_data = []
    for col in categorical_columns:
        unique_count = df[col].nunique()
        cardinality_data.append({
            'column': col,
            'unique_count': unique_count,
            'cardinality_level': '높음' if unique_count > 50 else '보통' if unique_count > 10 else '낮음'
        })
    
    cardinality_df = pd.DataFrame(cardinality_data)
    
    plt.subplot(2, 1, 1)
    plt.bar(cardinality_df['column'], cardinality_df['unique_count'])
    plt.title('범주형 변수별 고유값 개수')
    plt.ylabel('고유값 개수')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 1, 2)
    cardinality_counts = cardinality_df['cardinality_level'].value_counts()
    plt.pie(cardinality_counts.values, labels=cardinality_counts.index, autopct='%1.1f%%')
    plt.title('카디널리티 레벨 분포')
    
    plt.tight_layout()
    cardinality_path = "generated_plots/categorical_cardinality.png"
    plt.savefig(cardinality_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(cardinality_path)
    
    # 6. 결측값 패턴 분석
    missing_data = []
    for col in categorical_columns:
        missing_count = df[col].isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100
        missing_data.append({
            'column': col,
            'missing_count': missing_count,
            'missing_percentage': missing_percentage
        })
    
    missing_df = pd.DataFrame(missing_data)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(missing_df['column'], missing_df['missing_count'])
    plt.title('범주형 변수별 결측값 개수')
    plt.ylabel('결측값 개수')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.bar(missing_df['column'], missing_df['missing_percentage'])
    plt.title('범주형 변수별 결측값 비율')
    plt.ylabel('결측값 비율 (%)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    missing_path = "generated_plots/categorical_missing.png"
    plt.savefig(missing_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(missing_path)
    
    print(f"범주형 데이터 시각화 완료: {len(image_paths)}개 이미지 생성")
    
    return {
        **inputs,
        "categorical_images": image_paths
    }


# LangGraph 노드로 사용할 수 있는 함수
categorical_image_agent = RunnableLambda(create_categorical_visualizations)
