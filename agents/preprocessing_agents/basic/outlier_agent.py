"""
이상치 처리 전처리 에이전트
데이터의 이상치를 탐지하고 다양한 방법으로 처리합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def handle_outliers(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    이상치를 탐지하고 처리하는 전처리 함수
    
    Args:
        inputs: DataFrame과 처리 방법이 포함된 입력 딕셔너리
        
    Returns:
        이상치가 처리된 DataFrame이 포함된 딕셔너리
    """
    df = inputs["dataframe"].copy()
    method = inputs.get("outlier_method", "auto")  # auto, iqr, zscore, isolation_forest
    action = inputs.get("outlier_action", "cap")  # cap, remove, winsorize
    
    # EDA 결과물들 가져오기
    outlier_analysis_text = inputs.get("outlier_analysis_text", "")
    outlier_image_paths = inputs.get("outlier_image_paths", [])
    numeric_analysis_text = inputs.get("numeric_analysis_text", "")
    text_analysis = inputs.get("text_analysis", "")
    
    # 수치형 컬럼만 선택
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_columns:
        print("이상치 처리: 수치형 변수가 없습니다.")
        return {
            **inputs,
            "dataframe": df,
            "outlier_info": {}
        }
    
    outlier_info = {}
    print(f"이상치 처리 시작: {method} 방법, {action} 액션")
    
    # MultiModal LLM을 사용한 전처리 코드 생성
    if method == "auto":
        preprocessing_code = generate_outlier_code_with_llm(
            df, numeric_columns, outlier_analysis_text, outlier_image_paths,
            numeric_analysis_text, text_analysis, action
        )
        
        # 생성된 코드 실행
        try:
            exec(preprocessing_code)
            print("LLM 생성 코드로 이상치 처리 완료")
        except Exception as e:
            print(f"LLM 생성 코드 실행 오류: {e}")
            # 폴백: 기본 자동 처리
            df = apply_basic_outlier_handling(df, numeric_columns, action)
    else:
        # 수동 방법 사용
        df = apply_manual_outlier_handling(df, numeric_columns, method, action, inputs)
    
    # 처리 결과 요약
    total_outliers = sum(info.get('outliers_detected', 0) for info in outlier_info.values())
    total_removed = sum(info.get('outliers_removed', 0) for info in outlier_info.values())
    total_capped = sum(info.get('outliers_capped', 0) for info in outlier_info.values())
    
    print(f"이상치 처리 완료: 탐지 {total_outliers}개, 제거 {total_removed}개, 제한 {total_capped}개")
    
    return {
        **inputs,
        "dataframe": df,
        "outlier_info": outlier_info
    }


def generate_outlier_code_with_llm(df: pd.DataFrame, numeric_columns: List[str],
                                 outlier_analysis_text: str, outlier_image_paths: List[str],
                                 numeric_analysis_text: str, text_analysis: str, action: str) -> str:
    """
    MultiModal LLM을 사용하여 이상치 처리 코드를 생성합니다.
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
        outlier_info = f"{col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, range=[{stats['min']:.2f}, {stats['max']:.2f}]"
        numeric_summary.append(outlier_info)
    
    prompt = f"""
You are a data preprocessing expert. Please write Python code to handle outliers based on the following information.

=== Dataset Information ===
- Data size: {df.shape[0]} rows x {df.shape[1]} columns
- Numeric columns: {numeric_columns}
- Dataset head:
{df.head().to_string()}

=== Numeric Variable Statistics ===
{chr(10).join(numeric_summary)}

=== Outlier Analysis Results ===
{outlier_analysis_text}

=== Numeric Variable Analysis ===
{numeric_analysis_text}

=== Overall Data Analysis ===
{text_analysis}

=== Processing Method ===
- Action: {action} (cap=limit, remove=delete, winsorize=winsorization)

=== Requirements ===
1. Detect outliers using IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
2. Choose appropriate processing method based on distribution characteristics
3. Be cautious when outlier ratio is high
4. Code must be executable

Please write code in the following format:
```python
# Outlier handling code
# df is an already defined DataFrame
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


def apply_basic_outlier_handling(df: pd.DataFrame, numeric_columns: List[str], action: str) -> pd.DataFrame:
    """
    기본 이상치 처리 방법을 적용합니다.
    """
    for col in numeric_columns:
        # IQR 방법으로 이상치 탐지
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            print(f"  {col}: {outlier_count}개 이상치 탐지")
            
            if action == "cap":
                # 이상치를 상한/하한값으로 제한
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
                print(f"    → 이상치 {outlier_count}개 제한")
            elif action == "remove":
                # 이상치 제거
                df = df[~outliers]
                print(f"    → 이상치 {outlier_count}개 제거")
    
    return df


def apply_manual_outlier_handling(df: pd.DataFrame, numeric_columns: List[str], 
                                method: str, action: str, inputs: Dict) -> pd.DataFrame:
    """
    수동 이상치 처리 방법을 적용합니다.
    """
    for col in numeric_columns:
        outlier_info = {
            'original_count': len(df),
            'outliers_detected': 0,
            'outliers_removed': 0,
            'outliers_capped': 0
        }
        
        if method == "iqr":
            outliers = detect_outliers_iqr(df[col])
        elif method == "zscore":
            z_threshold = inputs.get("z_threshold", 3)
            outliers = detect_outliers_zscore(df[col], z_threshold)
        elif method == "isolation_forest":
            outliers = detect_outliers_isolation_forest(df[col])
        else:
            outliers = pd.Series([False] * len(df))
        
        outlier_count = outliers.sum()
        outlier_info['outliers_detected'] = outlier_count
        
        if outlier_count > 0:
            print(f"  {col}: {outlier_count}개 이상치 탐지")
            
            if action == "remove":
                # 이상치 제거
                df = df[~outliers]
                outlier_info['outliers_removed'] = outlier_count
                print(f"    → 이상치 {outlier_count}개 제거")
            
            elif action == "cap":
                # 이상치를 상한/하한값으로 제한
                if method in ["iqr", "auto"]:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                elif method == "zscore":
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    z_threshold = inputs.get("z_threshold", 3)
                    lower_bound = mean_val - z_threshold * std_val
                    upper_bound = mean_val + z_threshold * std_val
                else:
                    # 기본값
                    lower_bound = df[col].quantile(0.01)
                    upper_bound = df[col].quantile(0.99)
                
                # 이상치 제한
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
                outlier_info['outliers_capped'] = outlier_count
                print(f"    → 이상치 {outlier_count}개 제한 (범위: {lower_bound:.2f} ~ {upper_bound:.2f})")
            
            elif action == "winsorize":
                # Winsorization (상위/하위 1% 제한)
                lower_percentile = inputs.get("lower_percentile", 1)
                upper_percentile = inputs.get("upper_percentile", 99)
                
                lower_bound = df[col].quantile(lower_percentile / 100)
                upper_bound = df[col].quantile(upper_percentile / 100)
                
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
                outlier_info['outliers_capped'] = outlier_count
                print(f"    → Winsorization 적용 ({lower_percentile}% ~ {upper_percentile}%)")
    
    return df


def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """
    IQR 방법으로 이상치 탐지
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return (series < lower_bound) | (series > upper_bound)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3) -> pd.Series:
    """
    Z-score 방법으로 이상치 탐지
    """
    z_scores = np.abs((series - series.mean()) / series.std())
    return z_scores > threshold


def detect_outliers_isolation_forest(series: pd.Series) -> pd.Series:
    """
    Isolation Forest 방법으로 이상치 탐지
    """
    try:
        from sklearn.ensemble import IsolationForest
        
        # 2D 배열로 변환
        X = series.values.reshape(-1, 1)
        
        # Isolation Forest 모델 학습
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        predictions = iso_forest.fit_predict(X)
        
        # -1이 이상치
        return pd.Series(predictions == -1, index=series.index)
    
    except ImportError:
        print("  scikit-learn이 설치되지 않아 IQR 방법으로 대체합니다.")
        return detect_outliers_iqr(series)


# LangGraph 노드로 사용할 수 있는 함수
outlier_agent = RunnableLambda(handle_outliers)