"""
특성 선택 전처리 에이전트
중요한 특성만 선택하여 차원을 줄이는 다양한 방법을 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableLambda


def select_features(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    특성 선택을 수행하는 전처리 함수
    
    Args:
        inputs: DataFrame과 선택 방법이 포함된 입력 딕셔너리
        
    Returns:
        선택된 특성만 포함된 DataFrame이 포함된 딕셔너리
    """
    df = inputs["dataframe"].copy()
    method = inputs.get("selection_method", "auto")  # auto, variance, correlation, mutual_info, lasso, recursive
    target_column = inputs.get("target_column", None)
    n_features = inputs.get("n_features", None)  # 선택할 특성 수
    threshold = inputs.get("selection_threshold", 0.01)  # 임계값
    
    # 타겟 변수 분리
    if target_column and target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df
        y = None
    
    original_features = X.columns.tolist()
    print(f"특성 선택 시작: {method} 방법")
    print(f"  원본 특성 수: {len(original_features)}")
    
    if n_features:
        print(f"  목표 특성 수: {n_features}")
    
    selected_features = []
    selection_info = {
        'original_features': original_features,
        'method': method,
        'threshold': threshold
    }
    
    if method == "auto":
        # 자동 선택: 데이터 크기에 따라 결정
        if len(original_features) > 50:
            # 특성이 많은 경우 상관관계 기반 선택
            selected_features = select_by_correlation(X, threshold)
            method = "correlation"
        else:
            # 특성이 적은 경우 분산 기반 선택
            selected_features = select_by_variance(X, threshold)
            method = "variance"
    
    elif method == "variance":
        # 분산 기반 선택
        selected_features = select_by_variance(X, threshold)
    
    elif method == "correlation":
        # 상관관계 기반 선택
        selected_features = select_by_correlation(X, threshold)
    
    elif method == "mutual_info":
        # 상호정보량 기반 선택
        if y is not None:
            selected_features = select_by_mutual_info(X, y, n_features)
        else:
            print("  ⚠️  상호정보량 선택을 위한 타겟 변수가 없어 분산 기반으로 대체")
            selected_features = select_by_variance(X, threshold)
            method = "variance"
    
    elif method == "lasso":
        # Lasso 기반 선택
        if y is not None:
            selected_features = select_by_lasso(X, y, threshold)
        else:
            print("  ⚠️  Lasso 선택을 위한 타겟 변수가 없어 분산 기반으로 대체")
            selected_features = select_by_variance(X, threshold)
            method = "variance"
    
    elif method == "recursive":
        # 순환적 특성 제거
        if y is not None:
            selected_features = select_by_recursive(X, y, n_features)
        else:
            print("  ⚠️  순환적 선택을 위한 타겟 변수가 없어 분산 기반으로 대체")
            selected_features = select_by_variance(X, threshold)
            method = "variance"
    
    # 선택된 특성으로 데이터프레임 생성
    if selected_features:
        if target_column and target_column in df.columns:
            df_selected = df[selected_features + [target_column]]
        else:
            df_selected = df[selected_features]
        
        selection_info.update({
            'selected_features': selected_features,
            'final_method': method,
            'selected_count': len(selected_features),
            'removed_count': len(original_features) - len(selected_features)
        })
        
        print(f"  선택된 특성: {len(selected_features)}개")
        print(f"  제거된 특성: {len(original_features) - len(selected_features)}개")
        
        return {
            **inputs,
            "dataframe": df_selected,
            "selection_info": selection_info
        }
    else:
        print("  ⚠️  선택된 특성이 없어 원본 데이터를 유지합니다.")
        return {
            **inputs,
            "dataframe": df,
            "selection_info": selection_info
        }


def select_by_variance(X: pd.DataFrame, threshold: float) -> List[str]:
    """
    분산 기반 특성 선택
    
    Args:
        X: 특성 데이터프레임
        threshold: 분산 임계값
        
    Returns:
        선택된 특성 리스트
    """
    variances = X.var()
    selected = variances[variances > threshold].index.tolist()
    print(f"    → 분산 기반 선택: {len(selected)}개 특성 선택 (임계값: {threshold})")
    return selected


def select_by_correlation(X: pd.DataFrame, threshold: float) -> List[str]:
    """
    상관관계 기반 특성 선택
    
    Args:
        X: 특성 데이터프레임
        threshold: 상관관계 임계값
        
    Returns:
        선택된 특성 리스트
    """
    # 수치형 컬럼만 선택
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return numeric_cols
    
    # 상관관계 행렬 계산
    corr_matrix = X[numeric_cols].corr().abs()
    
    # 상관관계가 높은 특성 쌍 찾기
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # 상관관계가 임계값을 초과하는 특성 찾기
    high_corr_features = []
    for column in upper_triangle.columns:
        if (upper_triangle[column] > threshold).any():
            high_corr_features.append(column)
    
    # 상관관계가 높은 특성 중 하나만 선택
    selected = [col for col in numeric_cols if col not in high_corr_features]
    
    print(f"    → 상관관계 기반 선택: {len(selected)}개 특성 선택 (임계값: {threshold})")
    return selected


def select_by_mutual_info(X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
    """
    상호정보량 기반 특성 선택
    
    Args:
        X: 특성 데이터프레임
        y: 타겟 변수
        n_features: 선택할 특성 수
        
    Returns:
        선택된 특성 리스트
    """
    try:
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        
        # 수치형 컬럼만 선택
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            return []
        
        # 타겟 변수 타입에 따라 적절한 함수 선택
        if y.dtype in ['object', 'category']:
            mi_scores = mutual_info_classif(X[numeric_cols], y, random_state=42)
        else:
            mi_scores = mutual_info_regression(X[numeric_cols], y, random_state=42)
        
        # 상호정보량이 높은 순으로 정렬
        feature_scores = list(zip(numeric_cols, mi_scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 n_features개 선택
        if n_features:
            selected = [feature for feature, score in feature_scores[:n_features]]
        else:
            # 점수가 0보다 큰 특성만 선택
            selected = [feature for feature, score in feature_scores if score > 0]
        
        print(f"    → 상호정보량 기반 선택: {len(selected)}개 특성 선택")
        return selected
    
    except ImportError:
        print("    ⚠️  scikit-learn이 설치되지 않아 분산 기반으로 대체")
        return select_by_variance(X, 0.01)


def select_by_lasso(X: pd.DataFrame, y: pd.Series, threshold: float) -> List[str]:
    """
    Lasso 기반 특성 선택
    
    Args:
        X: 특성 데이터프레임
        y: 타겟 변수
        threshold: 계수 임계값
        
    Returns:
        선택된 특성 리스트
    """
    try:
        from sklearn.linear_model import Lasso
        
        # 수치형 컬럼만 선택
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            return []
        
        # Lasso 모델 학습
        lasso = Lasso(alpha=0.01, random_state=42)
        lasso.fit(X[numeric_cols], y)
        
        # 계수가 임계값보다 큰 특성 선택
        selected = [col for col in numeric_cols if abs(lasso.coef_[numeric_cols.index(col)]) > threshold]
        
        print(f"    → Lasso 기반 선택: {len(selected)}개 특성 선택 (임계값: {threshold})")
        return selected
    
    except ImportError:
        print("    ⚠️  scikit-learn이 설치되지 않아 분산 기반으로 대체")
        return select_by_variance(X, 0.01)


def select_by_recursive(X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
    """
    순환적 특성 제거
    
    Args:
        X: 특성 데이터프레임
        y: 타겟 변수
        n_features: 선택할 특성 수
        
    Returns:
        선택된 특성 리스트
    """
    try:
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LinearRegression
        
        # 수치형 컬럼만 선택
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            return []
        
        # RFE 모델 생성
        estimator = LinearRegression()
        rfe = RFE(estimator=estimator, n_features_to_select=n_features or len(numeric_cols)//2)
        
        # 특성 선택 수행
        rfe.fit(X[numeric_cols], y)
        
        # 선택된 특성
        selected = [col for col, selected in zip(numeric_cols, rfe.support_) if selected]
        
        print(f"    → 순환적 특성 제거: {len(selected)}개 특성 선택")
        return selected
    
    except ImportError:
        print("    ⚠️  scikit-learn이 설치되지 않아 분산 기반으로 대체")
        return select_by_variance(X, 0.01)


# LangGraph 노드로 사용할 수 있는 함수
feature_selection_agent = RunnableLambda(select_features)