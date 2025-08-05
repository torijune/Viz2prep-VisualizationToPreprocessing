"""
변수별 상관관계 분석 이미지 에이전트
상관관계 히트맵, 산점도, 상관관계 네트워크 등을 생성하여 시각적 분석 결과를 제공합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from langchain_core.runnables import RunnableLambda
import os


def create_correlation_visualizations(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    변수 간 상관관계의 시각적 분석을 수행하고 이미지를 생성합니다.
    
    Args:
        inputs: DataFrame이 포함된 입력 딕셔너리
        
    Returns:
        생성된 이미지 경로들이 포함된 딕셔너리
    """
    df = inputs["dataframe"]
    
    # 수치형 컬럼만 선택
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 2:
        return {
            **inputs,
            "correlation_images": []
        }
    
    # 이미지 저장 디렉토리 생성
    os.makedirs("generated_plots", exist_ok=True)
    
    image_paths = []
    
    # 1. 상관관계 히트맵
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numeric_columns].corr()
    
    # 히트맵 생성
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
               square=True, linewidths=0.5, fmt='.2f')
    plt.title('변수 간 상관관계 히트맵')
    plt.tight_layout()
    corr_heatmap_path = "generated_plots/correlation_heatmap.png"
    plt.savefig(corr_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(corr_heatmap_path)
    
    # 2. 타겟 변수와의 상관관계 분석
    target_column = None
    for col in ['Survived', 'target', 'label', 'class', 'y']:
        if col in numeric_columns:
            target_column = col
            break
    
    if target_column:
        plt.figure(figsize=(12, 8))
        
        # 타겟 변수와의 상관관계
        target_correlations = correlation_matrix[target_column].sort_values(key=abs, ascending=False)
        target_correlations = target_correlations[target_correlations.index != target_column]
        
        # 상관관계 막대그래프
        plt.subplot(2, 1, 1)
        colors = ['red' if x < 0 else 'blue' for x in target_correlations.values]
        plt.bar(range(len(target_correlations)), target_correlations.values, color=colors)
        plt.title(f'{target_column}와의 상관관계')
        plt.ylabel('상관계수')
        plt.xticks(range(len(target_correlations)), target_correlations.index, rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 절댓값 기준 정렬
        plt.subplot(2, 1, 2)
        abs_correlations = target_correlations.abs().sort_values(ascending=False)
        colors = ['red' if target_correlations[col] < 0 else 'blue' for col in abs_correlations.index]
        plt.bar(range(len(abs_correlations)), abs_correlations.values, color=colors)
        plt.title(f'{target_column}와의 상관관계 (절댓값 기준)')
        plt.ylabel('|상관계수|')
        plt.xticks(range(len(abs_correlations)), abs_correlations.index, rotation=45, ha='right')
        
        plt.tight_layout()
        target_corr_path = "generated_plots/target_correlation.png"
        plt.savefig(target_corr_path, dpi=300, bbox_inches='tight')
        plt.close()
        image_paths.append(target_corr_path)
    
    # 3. 산점도 매트릭스 (상위 6개 변수)
    if len(numeric_columns) <= 6:
        plot_columns = numeric_columns
    else:
        # 상관관계가 높은 변수들 우선 선택
        if target_column:
            target_corr_abs = correlation_matrix[target_column].abs().sort_values(ascending=False)
            top_vars = target_corr_abs.head(6).index.tolist()
            if target_column not in top_vars:
                top_vars = [target_column] + top_vars[:5]
        else:
            # 전체 상관관계 평균이 높은 변수들 선택
            mean_corr = correlation_matrix.abs().mean().sort_values(ascending=False)
            top_vars = mean_corr.head(6).index.tolist()
        plot_columns = top_vars
    
    if len(plot_columns) >= 2:
        plt.figure(figsize=(15, 12))
        sns.pairplot(df[plot_columns], diag_kind='hist')
        plt.suptitle('변수 간 산점도 매트릭스', y=1.02)
        plt.tight_layout()
        pairplot_path = "generated_plots/correlation_pairplot.png"
        plt.savefig(pairplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        image_paths.append(pairplot_path)
    
    # 4. 강한 상관관계 네트워크
    strong_correlations = []
    for i, col1 in enumerate(numeric_columns):
        for j, col2 in enumerate(numeric_columns[i+1:], i+1):
            corr_value = correlation_matrix.loc[col1, col2]
            if abs(corr_value) >= 0.5:
                strong_correlations.append({
                    'var1': col1,
                    'var2': col2,
                    'correlation': corr_value
                })
    
    if strong_correlations:
        plt.figure(figsize=(12, 8))
        
        # 네트워크 그래프 생성
        import networkx as nx
        
        G = nx.Graph()
        
        # 노드 추가
        all_vars = set()
        for corr in strong_correlations:
            all_vars.add(corr['var1'])
            all_vars.add(corr['var2'])
        
        for var in all_vars:
            G.add_node(var)
        
        # 엣지 추가
        for corr in strong_correlations:
            weight = abs(corr['correlation'])
            color = 'red' if corr['correlation'] < 0 else 'blue'
            G.add_edge(corr['var1'], corr['var2'], weight=weight, color=color)
        
        # 레이아웃 설정
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 노드 그리기
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        # 엣지 그리기
        edges = G.edges(data=True)
        edge_colors = [edge['color'] for edge in edges]
        edge_weights = [edge['weight'] * 3 for edge in edges]  # 가중치를 시각화
        
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_weights, alpha=0.7)
        
        plt.title('강한 상관관계 네트워크 (|r| >= 0.5)')
        plt.axis('off')
        plt.tight_layout()
        network_path = "generated_plots/correlation_network.png"
        plt.savefig(network_path, dpi=300, bbox_inches='tight')
        plt.close()
        image_paths.append(network_path)
    
    # 5. 상관관계 분포 히스토그램
    plt.figure(figsize=(12, 8))
    
    # 모든 상관관계 값 추출
    all_correlations = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
    
    plt.subplot(2, 2, 1)
    plt.hist(all_correlations, bins=20, alpha=0.7, edgecolor='black')
    plt.title('상관관계 분포')
    plt.xlabel('상관계수')
    plt.ylabel('빈도')
    
    plt.subplot(2, 2, 2)
    plt.hist(all_correlations[all_correlations > 0], bins=15, alpha=0.7, color='blue', edgecolor='black')
    plt.title('양의 상관관계 분포')
    plt.xlabel('상관계수')
    plt.ylabel('빈도')
    
    plt.subplot(2, 2, 3)
    plt.hist(all_correlations[all_correlations < 0], bins=15, alpha=0.7, color='red', edgecolor='black')
    plt.title('음의 상관관계 분포')
    plt.xlabel('상관계수')
    plt.ylabel('빈도')
    
    plt.subplot(2, 2, 4)
    abs_correlations = np.abs(all_correlations)
    plt.hist(abs_correlations, bins=15, alpha=0.7, color='green', edgecolor='black')
    plt.title('상관관계 강도 분포')
    plt.xlabel('|상관계수|')
    plt.ylabel('빈도')
    
    plt.tight_layout()
    corr_dist_path = "generated_plots/correlation_distribution.png"
    plt.savefig(corr_dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(corr_dist_path)
    
    # 6. 다중공선성 분석
    plt.figure(figsize=(12, 8))
    
    # 상관계수 0.8 이상인 변수 쌍
    high_corr_pairs = []
    for i, col1 in enumerate(numeric_columns):
        for j, col2 in enumerate(numeric_columns[i+1:], i+1):
            corr_value = correlation_matrix.loc[col1, col2]
            if abs(corr_value) >= 0.8:
                high_corr_pairs.append({
                    'var1': col1,
                    'var2': col2,
                    'correlation': corr_value
                })
    
    if high_corr_pairs:
        plt.subplot(2, 1, 1)
        pair_labels = [f"{pair['var1']}\n{pair['var2']}" for pair in high_corr_pairs]
        corr_values = [pair['correlation'] for pair in high_corr_pairs]
        colors = ['red' if x < 0 else 'blue' for x in corr_values]
        
        plt.bar(range(len(high_corr_pairs)), corr_values, color=colors)
        plt.title('다중공선성이 의심되는 변수 쌍 (|r| >= 0.8)')
        plt.ylabel('상관계수')
        plt.xticks(range(len(high_corr_pairs)), pair_labels, rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.subplot(2, 1, 2)
        abs_values = [abs(pair['correlation']) for pair in high_corr_pairs]
        plt.bar(range(len(high_corr_pairs)), abs_values, color='orange')
        plt.title('다중공선성 강도')
        plt.ylabel('|상관계수|')
        plt.xticks(range(len(high_corr_pairs)), pair_labels, rotation=45, ha='right')
    else:
        plt.text(0.5, 0.5, '다중공선성 문제 없음', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('다중공선성 분석')
    
    plt.tight_layout()
    multicollinearity_path = "generated_plots/multicollinearity.png"
    plt.savefig(multicollinearity_path, dpi=300, bbox_inches='tight')
    plt.close()
    image_paths.append(multicollinearity_path)
    
    print(f"상관관계 시각화 완료: {len(image_paths)}개 이미지 생성")
    
    return {
        **inputs,
        "correlation_images": image_paths
    }


# LangGraph 노드로 사용할 수 있는 함수
correlation_image_agent = RunnableLambda(create_correlation_visualizations)
