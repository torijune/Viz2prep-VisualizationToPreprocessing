"""
차원 축소 전처리 에이전트
고차원 데이터를 저차원으로 축소하는 다양한 방법을 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def reduce_dimensions(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    차원 축소를 수행하는 전처리 함수
    
    Args:
        inputs: DataFrame과 축소 방법이 포함된 입력 딕셔너리
        
    Returns:
        차원이 축소된 DataFrame이 포함된 딕셔너리
    """
    df = inputs["dataframe"].copy()
    method = inputs.get("dimension_method", "auto")  # auto, pca, tsne, umap, lda
    n_components = inputs.get("n_components", None)  # 축소할 차원 수
    target_column = inputs.get("target_column", None)
    
    # EDA 결과물들 가져오기
    text_analysis = inputs.get("text_analysis", "")
    corr_analysis_text = inputs.get("corr_analysis_text", "")
    numeric_analysis_text = inputs.get("numeric_analysis_text", "")
    
    # 수치형 컬럼만 선택
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_column and target_column in numeric_columns:
        numeric_columns.remove(target_column)
    
    if len(numeric_columns) < 2:
        print("차원 축소: 수치형 변수가 2개 미만입니다.")
        return {
            **inputs,
            "dataframe": df,
            "dimension_info": {}
        }
    
    # 자동으로 차원 수 결정
    if n_components is None:
        n_components = min(len(numeric_columns), 10)  # 최대 10개 차원으로 제한
    
    print(f"차원 축소 시작: {method} 방법")
    print(f"  원본 차원: {len(numeric_columns)}")
    print(f"  목표 차원: {n_components}")
    
    # MultiModal LLM을 사용한 전처리 코드 생성
    if method == "auto":
        preprocessing_code = generate_dimension_reduction_code_with_llm(
            df, numeric_columns, target_column, n_components,
            text_analysis, corr_analysis_text, numeric_analysis_text
        )
        
        # 생성된 코드 실행
        try:
            exec(preprocessing_code)
            print("LLM 생성 코드로 차원 축소 완료")
        except Exception as e:
            print(f"LLM 생성 코드 실행 오류: {e}")
            # 폴백: 기본 자동 처리
            df = apply_basic_dimension_reduction(df, numeric_columns, target_column, n_components)
    else:
        # 수동 방법 사용
        df = apply_manual_dimension_reduction(df, numeric_columns, target_column, method, n_components, inputs)
    
    # 축소 결과 정보
    dimension_info = {
        'original_dimensions': len(numeric_columns),
        'target_dimensions': n_components,
        'method': method,
        'numeric_columns': numeric_columns,
        'final_method': method,
        'reduced_columns': [col for col in df.columns if col not in numeric_columns and col != target_column],
        'variance_explained': get_variance_explained(df[numeric_columns], df) if method == "pca" else None
    }
    
    print(f"차원 축소 완료: {len(dimension_info['reduced_columns'])}개 차원으로 축소")
    
    return {
        **inputs,
        "dataframe": df,
        "dimension_info": dimension_info
    }


def generate_dimension_reduction_code_with_llm(df: pd.DataFrame, numeric_columns: List[str],
                                            target_column: Optional[str], n_components: int,
                                            text_analysis: str, corr_analysis_text: str,
                                            numeric_analysis_text: str) -> str:
    """
    MultiModal LLM을 사용하여 차원 축소 코드를 생성합니다.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=2000
    )
    
    # 수치형 변수 정보 요약
    numeric_summary = []
    for col in numeric_columns:
        stats = df[col].describe()
        summary = f"{col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, range=[{stats['min']:.2f}, {stats['max']:.2f}]"
        numeric_summary.append(summary)
    
    prompt = f"""
You are a data preprocessing expert. Please write Python code to perform dimensionality reduction based on the following information.

=== Dataset Information ===
- Data size: {df.shape[0]} rows x {df.shape[1]} columns
- Numeric columns: {numeric_columns}
- Target variable: {target_column or 'None'}
- Target dimensions: {n_components}
- Dataset head:
{df.head().to_string()}

=== Numeric Variable Statistics ===
{chr(10).join(numeric_summary)}

=== Overall Data Analysis ===
{text_analysis}

=== Correlation Analysis ===
{corr_analysis_text}

=== Numeric Variable Analysis ===
{numeric_analysis_text}

=== Requirements ===
1. Choose appropriate method based on data size:
   - High-dimensional data (>50): PCA
   - Low-dimensional data: t-SNE
2. Consider LDA if target variable exists
3. Code must be executable

Please write code in the following format:
```python
# Dimensionality reduction code
# df is an already defined DataFrame
# Update df with reduced features
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


def apply_basic_dimension_reduction(df: pd.DataFrame, numeric_columns: List[str],
                                 target_column: Optional[str], n_components: int) -> pd.DataFrame:
    """
    기본 차원 축소 방법을 적용합니다.
    """
    # 자동 선택: 데이터 크기에 따라 결정
    if len(numeric_columns) > 50:
        # 고차원 데이터: PCA 사용
        df_reduced = apply_pca(df[numeric_columns], n_components)
        method = "pca"
    else:
        # 저차원 데이터: t-SNE 사용
        df_reduced = apply_tsne(df[numeric_columns], n_components)
        method = "tsne"
    
    # 축소된 특성과 원본 특성 결합
    if target_column and target_column in df.columns:
        df_final = pd.concat([df_reduced, df[target_column]], axis=1)
    else:
        df_final = df_reduced
    
    print(f"  {method} 방법으로 차원 축소 완료")
    return df_final


def apply_manual_dimension_reduction(df: pd.DataFrame, numeric_columns: List[str],
                                  target_column: Optional[str], method: str,
                                  n_components: int, inputs: Dict) -> pd.DataFrame:
    """
    수동 차원 축소 방법을 적용합니다.
    """
    if method == "pca":
        df_reduced = apply_pca(df[numeric_columns], n_components)
    elif method == "tsne":
        df_reduced = apply_tsne(df[numeric_columns], n_components)
    elif method == "umap":
        df_reduced = apply_umap(df[numeric_columns], n_components)
    elif method == "lda":
        if target_column and target_column in df.columns:
            df_reduced = apply_lda(df[numeric_columns], df[target_column], n_components)
        else:
            print("  ⚠️  LDA를 위한 타겟 변수가 없어 PCA로 대체")
            df_reduced = apply_pca(df[numeric_columns], n_components)
    else:
        print(f"  ⚠️  알 수 없는 방법 '{method}'입니다. PCA를 사용합니다.")
        df_reduced = apply_pca(df[numeric_columns], n_components)
    
    # 축소된 특성과 원본 특성 결합
    if target_column and target_column in df.columns:
        df_final = pd.concat([df_reduced, df[target_column]], axis=1)
    else:
        df_final = df_reduced
    
    return df_final


def apply_pca(X: pd.DataFrame, n_components: int) -> pd.DataFrame:
    """
    PCA를 적용하여 차원 축소
    """
    try:
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=n_components, random_state=42)
        X_reduced = pca.fit_transform(X)
        
        # 컬럼명 생성
        column_names = [f"PC_{i+1}" for i in range(n_components)]
        df_reduced = pd.DataFrame(X_reduced, columns=column_names, index=X.index)
        
        print(f"    → PCA 적용: 분산 설명률 {sum(pca.explained_variance_ratio_):.3f}")
        return df_reduced
    
    except ImportError:
        print("    ⚠️  scikit-learn이 설치되지 않아 원본 데이터를 유지합니다.")
        return X


def apply_tsne(X: pd.DataFrame, n_components: int) -> pd.DataFrame:
    """
    t-SNE를 적용하여 차원 축소
    """
    try:
        from sklearn.manifold import TSNE
        
        # 데이터가 너무 크면 샘플링
        if len(X) > 5000:
            print("    ⚠️  데이터가 너무 커서 5000개 샘플로 제한합니다.")
            sample_indices = np.random.choice(len(X), 5000, replace=False)
            X_sampled = X.iloc[sample_indices]
        else:
            X_sampled = X
        
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(X_sampled)-1))
        X_reduced = tsne.fit_transform(X_sampled)
        
        # 컬럼명 생성
        column_names = [f"tSNE_{i+1}" for i in range(n_components)]
        df_reduced = pd.DataFrame(X_reduced, columns=column_names, index=X_sampled.index)
        
        print(f"    → t-SNE 적용: {len(X_sampled)}개 샘플")
        return df_reduced
    
    except ImportError:
        print("    ⚠️  scikit-learn이 설치되지 않아 PCA로 대체")
        return apply_pca(X, n_components)


def apply_umap(X: pd.DataFrame, n_components: int) -> pd.DataFrame:
    """
    UMAP을 적용하여 차원 축소
    """
    try:
        import umap
        
        # 데이터가 너무 크면 샘플링
        if len(X) > 10000:
            print("    ⚠️  데이터가 너무 커서 10000개 샘플로 제한합니다.")
            sample_indices = np.random.choice(len(X), 10000, replace=False)
            X_sampled = X.iloc[sample_indices]
        else:
            X_sampled = X
        
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        X_reduced = reducer.fit_transform(X_sampled)
        
        # 컬럼명 생성
        column_names = [f"UMAP_{i+1}" for i in range(n_components)]
        df_reduced = pd.DataFrame(X_reduced, columns=column_names, index=X_sampled.index)
        
        print(f"    → UMAP 적용: {len(X_sampled)}개 샘플")
        return df_reduced
    
    except ImportError:
        print("    ⚠️  umap-learn이 설치되지 않아 PCA로 대체")
        return apply_pca(X, n_components)


def apply_lda(X: pd.DataFrame, y: pd.Series, n_components: int) -> pd.DataFrame:
    """
    LDA를 적용하여 차원 축소
    """
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        
        # 클래스 수에 따라 차원 수 조정
        n_classes = len(y.unique())
        n_components = min(n_components, n_classes - 1)
        
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_reduced = lda.fit_transform(X, y)
        
        # 컬럼명 생성
        column_names = [f"LDA_{i+1}" for i in range(n_components)]
        df_reduced = pd.DataFrame(X_reduced, columns=column_names, index=X.index)
        
        print(f"    → LDA 적용: {n_components}개 차원")
        return df_reduced
    
    except ImportError:
        print("    ⚠️  scikit-learn이 설치되지 않아 PCA로 대체")
        return apply_pca(X, n_components)


def get_variance_explained(X_original: pd.DataFrame, X_reduced: pd.DataFrame) -> float:
    """
    PCA의 분산 설명률 계산
    """
    try:
        from sklearn.decomposition import PCA
        
        pca = PCA()
        pca.fit(X_original)
        
        return sum(pca.explained_variance_ratio_[:len(X_reduced.columns)])
    
    except:
        return None


# LangGraph 노드로 사용할 수 있는 함수
dimension_agent = RunnableLambda(reduce_dimensions)