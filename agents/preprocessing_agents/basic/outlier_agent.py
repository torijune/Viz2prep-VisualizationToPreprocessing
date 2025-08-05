"""
이상치 처리 전처리 에이전트
데이터의 이상치를 탐지하고 다양한 방법으로 처리합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableLambda


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
    
    for col in numeric_columns:
        outlier_info[col] = {
            'original_count': len(df),
            'outliers_detected': 0,
            'outliers_removed': 0,
            'outliers_capped': 0
        }
        
        if method == "auto":
            # 자동 선택: IQR 방법 사용
            outliers = detect_outliers_iqr(df[col])
        elif method == "iqr":
            outliers = detect_outliers_iqr(df[col])
        elif method == "zscore":
            z_threshold = inputs.get("z_threshold", 3)
            outliers = detect_outliers_zscore(df[col], z_threshold)
        elif method == "isolation_forest":
            outliers = detect_outliers_isolation_forest(df[col])
        else:
            outliers = pd.Series([False] * len(df))
        
        outlier_count = outliers.sum()
        outlier_info[col]['outliers_detected'] = outlier_count
        
        if outlier_count > 0:
            print(f"  {col}: {outlier_count}개 이상치 탐지")
            
            if action == "remove":
                # 이상치 제거
                df = df[~outliers]
                outlier_info[col]['outliers_removed'] = outlier_count
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
                outlier_info[col]['outliers_capped'] = outlier_count
                print(f"    → 이상치 {outlier_count}개 제한 (범위: {lower_bound:.2f} ~ {upper_bound:.2f})")
            
            elif action == "winsorize":
                # Winsorization (상위/하위 1% 제한)
                lower_percentile = inputs.get("lower_percentile", 1)
                upper_percentile = inputs.get("upper_percentile", 99)
                
                lower_bound = df[col].quantile(lower_percentile / 100)
                upper_bound = df[col].quantile(upper_percentile / 100)
                
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
                outlier_info[col]['outliers_capped'] = outlier_count
                print(f"    → Winsorization 적용 ({lower_percentile}% ~ {upper_percentile}%)")
    
    # 처리 결과 요약
    total_outliers = sum(info['outliers_detected'] for info in outlier_info.values())
    total_removed = sum(info['outliers_removed'] for info in outlier_info.values())
    total_capped = sum(info['outliers_capped'] for info in outlier_info.values())
    
    print(f"이상치 처리 완료: 탐지 {total_outliers}개, 제거 {total_removed}개, 제한 {total_capped}개")
    
    return {
        **inputs,
        "dataframe": df,
        "outlier_info": outlier_info
    }


def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """
    IQR 방법으로 이상치 탐지
    
    Args:
        series: 탐지할 시리즈
        
    Returns:
        이상치 여부를 나타내는 불린 시리즈
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
    
    Args:
        series: 탐지할 시리즈
        threshold: Z-score 임계값
        
    Returns:
        이상치 여부를 나타내는 불린 시리즈
    """
    z_scores = np.abs((series - series.mean()) / series.std())
    return z_scores > threshold


def detect_outliers_isolation_forest(series: pd.Series) -> pd.Series:
    """
    Isolation Forest 방법으로 이상치 탐지
    
    Args:
        series: 탐지할 시리즈
        
    Returns:
        이상치 여부를 나타내는 불린 시리즈
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