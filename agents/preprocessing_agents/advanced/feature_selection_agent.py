"""
특성 선택 전처리 에이전트
중요한 특성만 선택하여 차원을 줄이는 다양한 방법을 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


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
    
    # EDA 결과물들 가져오기
    corr_analysis_text = inputs.get("corr_analysis_text", "")
    corr_image_paths = inputs.get("corr_image_paths", [])
    numeric_analysis_text = inputs.get("numeric_analysis_text", "")
    text_analysis = inputs.get("text_analysis", "")
    
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
    
    # MultiModal LLM을 사용한 전처리 코드 생성
    if method == "auto":
        preprocessing_code = generate_feature_selection_code_with_llm(
            df, X, y, original_features, corr_analysis_text, corr_image_paths,
            numeric_analysis_text, text_analysis, n_features, threshold
        )
        
        # 생성된 코드 실행
        try:
            exec(preprocessing_code)
            print("LLM 생성 코드로 특성 선택 완료")
        except Exception as e:
            print(f"LLM 생성 코드 실행 오류: {e}")
            # 폴백: 기본 자동 처리
            selected_features = apply_basic_feature_selection(X, threshold)
            if selected_features:
                if target_column and target_column in df.columns:
                    df_selected = df[selected_features + [target_column]]
                else:
                    df_selected = df[selected_features]
                df = df_selected
    else:
        # 수동 방법 사용
        selected_features = apply_manual_feature_selection(X, y, method, n_features, threshold)
        if selected_features:
            if target_column and target_column in df.columns:
                df_selected = df[selected_features + [target_column]]
            else:
                df_selected = df[selected_features]
            df = df_selected
    
    # 선택 결과 정보
    final_columns = df.columns.tolist()
    selection_info = {
        'original_features': original_features,
        'method': method,
        'threshold': threshold,
        'selected_features': final_columns,
        'final_method': method,
        'selected_count': len(final_columns),
        'removed_count': len(original_features) - len(final_columns)
    }
    
    print(f"  선택된 특성: {len(final_columns)}개")
    print(f"  제거된 특성: {len(original_features) - len(final_columns)}개")
    
    return {
        **inputs,
        "dataframe": df,
        "selection_info": selection_info
    }


def generate_feature_selection_code_with_llm(df: pd.DataFrame, X: pd.DataFrame, y: Optional[pd.Series],
                                          original_features: List[str], corr_analysis_text: str, 
                                          corr_image_paths: List[str], numeric_analysis_text: str,
                                          text_analysis: str, n_features: Optional[int], threshold: float) -> str:
    """
    MultiModal LLM을 사용하여 특성 선택 코드를 생성합니다.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=2000
    )
    
    # 특성 정보 요약
    feature_summary = []
    for col in original_features:
        if col in X.columns:
            if X[col].dtype in [np.number]:
                stats = X[col].describe()
                summary = f"{col}: numeric, mean={stats['mean']:.2f}, std={stats['std']:.2f}"
            else:
                unique_count = X[col].nunique()
                summary = f"{col}: categorical, {unique_count} unique values"
            feature_summary.append(summary)
    
    prompt = f"""
You are a data preprocessing expert. Please write Python code to perform feature selection based on the following information.

=== Dataset Information ===
- Data size: {df.shape[0]} rows x {df.shape[1]} columns
- Original features: {original_features}
- Target feature count: {n_features or 'automatic'}
- Dataset head:
{df.head().to_string()}

=== Feature Information ===
{chr(10).join(feature_summary)}

=== Correlation Analysis Results ===
{corr_analysis_text}

=== Numeric Variable Analysis ===
{numeric_analysis_text}

=== Overall Data Analysis ===
{text_analysis}

=== Requirements ===
1. Select only one from highly correlated variable pairs
2. Remove variables with low variance
3. Choose appropriate method based on number of features
4. Code must be executable

Please write code in the following format:
```python
# Feature selection code
# df is an already defined DataFrame
# Update df with selected features
```

Return only the code without explanations.
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        code = response.content
        
        # 코드 블록에서 실제 코드만 추출
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        
        return code
    except Exception as e:
        print(f"LLM 코드 생성 오류: {e}")
        return ""


def apply_basic_feature_selection(X: pd.DataFrame, threshold: float) -> List[str]:
    """
    기본 특성 선택 방법을 적용합니다.
    """
    # 분산 기반 선택
    variances = X.var()
    selected = variances[variances > threshold].index.tolist()
    print(f"    → 분산 기반 선택: {len(selected)}개 특성 선택 (임계값: {threshold})")
    return selected


def apply_manual_feature_selection(X: pd.DataFrame, y: Optional[pd.Series], 
                                 method: str, n_features: Optional[int], threshold: float) -> List[str]:
    """
    수동 특성 선택 방법을 적용합니다.
    """
    if method == "variance":
        return select_by_variance(X, threshold)
    elif method == "correlation":
        return select_by_correlation(X, threshold)
    elif method == "mutual_info":
        if y is not None:
            return select_by_mutual_info(X, y, n_features)
        else:
            print("  ⚠️  상호정보량 선택을 위한 타겟 변수가 없어 분산 기반으로 대체")
            return select_by_variance(X, threshold)
    elif method == "lasso":
        if y is not None:
            return select_by_lasso(X, y, threshold)
        else:
            print("  ⚠️  Lasso 선택을 위한 타겟 변수가 없어 분산 기반으로 대체")
            return select_by_variance(X, threshold)
    elif method == "recursive":
        if y is not None:
            return select_by_recursive(X, y, n_features)
        else:
            print("  ⚠️  순환적 선택을 위한 타겟 변수가 없어 분산 기반으로 대체")
            return select_by_variance(X, threshold)
    else:
        return select_by_variance(X, threshold)


def select_by_variance(X: pd.DataFrame, threshold: float) -> List[str]:
    """
    분산 기반 특성 선택
    """
    variances = X.var()
    selected = variances[variances > threshold].index.tolist()
    print(f"    → 분산 기반 선택: {len(selected)}개 특성 선택 (임계값: {threshold})")
    return selected


def select_by_correlation(X: pd.DataFrame, threshold: float) -> List[str]:
    """
    상관관계 기반 특성 선택
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