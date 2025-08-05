"""
클래스 불균형 처리 전처리 에이전트
타겟 변수의 클래스 불균형을 해결하는 다양한 방법을 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableLambda


def handle_imbalance(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    클래스 불균형을 처리하는 전처리 함수
    
    Args:
        inputs: DataFrame과 처리 방법이 포함된 입력 딕셔너리
        
    Returns:
        불균형이 처리된 DataFrame이 포함된 딕셔너리
    """
    df = inputs["dataframe"].copy()
    method = inputs.get("imbalance_method", "auto")  # auto, smote, adasyn, undersample, oversample
    target_column = inputs.get("target_column", None)
    
    if not target_column or target_column not in df.columns:
        print("클래스 불균형 처리: 타겟 변수가 지정되지 않았습니다.")
        return {
            **inputs,
            "dataframe": df,
            "imbalance_info": {}
        }
    
    # 타겟 변수 분리
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 클래스 분포 확인
    class_counts = y.value_counts()
    total_samples = len(y)
    imbalance_ratio = class_counts.min() / class_counts.max()
    
    print(f"클래스 불균형 처리 시작: {method} 방법")
    print(f"  타겟 변수: {target_column}")
    print(f"  클래스 분포: {dict(class_counts)}")
    print(f"  불균형 비율: {imbalance_ratio:.3f}")
    
    imbalance_info = {
        'target_column': target_column,
        'original_class_counts': dict(class_counts),
        'imbalance_ratio': imbalance_ratio,
        'method': method
    }
    
    # 불균형이 심한지 판단 (비율이 0.1 이하)
    is_imbalanced = imbalance_ratio < 0.1
    
    if not is_imbalanced:
        print("  클래스 불균형이 심하지 않습니다. 처리하지 않습니다.")
        return {
            **inputs,
            "dataframe": df,
            "imbalance_info": imbalance_info
        }
    
    if method == "auto":
        # 자동 선택: 불균형 비율에 따라 결정
        if imbalance_ratio < 0.05:
            # 매우 심한 불균형: SMOTE 사용
            X_resampled, y_resampled = apply_smote(X, y)
            method = "smote"
        elif imbalance_ratio < 0.1:
            # 중간 불균형: ADASYN 사용
            X_resampled, y_resampled = apply_adasyn(X, y)
            method = "adasyn"
        else:
            # 약한 불균형: 언더샘플링 사용
            X_resampled, y_resampled = apply_undersampling(X, y)
            method = "undersample"
    
    elif method == "smote":
        X_resampled, y_resampled = apply_smote(X, y)
    
    elif method == "adasyn":
        X_resampled, y_resampled = apply_adasyn(X, y)
    
    elif method == "undersample":
        X_resampled, y_resampled = apply_undersampling(X, y)
    
    elif method == "oversample":
        X_resampled, y_resampled = apply_oversampling(X, y)
    
    else:
        print(f"  ⚠️  알 수 없는 방법 '{method}'입니다. 원본 데이터를 유지합니다.")
        return {
            **inputs,
            "dataframe": df,
            "imbalance_info": imbalance_info
        }
    
    # 재샘플링된 데이터로 데이터프레임 재구성
    df_resampled = X_resampled.copy()
    df_resampled[target_column] = y_resampled
    
    # 처리 후 클래스 분포 확인
    new_class_counts = y_resampled.value_counts()
    new_imbalance_ratio = new_class_counts.min() / new_class_counts.max()
    
    imbalance_info.update({
        'final_method': method,
        'new_class_counts': dict(new_class_counts),
        'new_imbalance_ratio': new_imbalance_ratio,
        'samples_added': len(y_resampled) - total_samples
    })
    
    print(f"  처리 후 클래스 분포: {dict(new_class_counts)}")
    print(f"  처리 후 불균형 비율: {new_imbalance_ratio:.3f}")
    print(f"  추가된 샘플: {len(y_resampled) - total_samples}개")
    
    return {
        **inputs,
        "dataframe": df_resampled,
        "imbalance_info": imbalance_info
    }


def apply_smote(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    SMOTE를 적용하여 오버샘플링
    
    Args:
        X: 특성 데이터프레임
        y: 타겟 변수
        
    Returns:
        (재샘플링된 특성, 재샘플링된 타겟)
    """
    try:
        from imblearn.over_sampling import SMOTE
        
        # 수치형 컬럼만 선택
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            print("    ⚠️  수치형 특성이 없어 SMOTE를 적용할 수 없습니다.")
            return X, y
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X[numeric_cols], y)
        
        print(f"    → SMOTE 적용: {len(y_resampled)}개 샘플")
        return X_resampled, y_resampled
    
    except ImportError:
        print("    ⚠️  imbalanced-learn이 설치되지 않아 오버샘플링으로 대체")
        return apply_oversampling(X, y)


def apply_adasyn(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    ADASYN을 적용하여 오버샘플링
    
    Args:
        X: 특성 데이터프레임
        y: 타겟 변수
        
    Returns:
        (재샘플링된 특성, 재샘플링된 타겟)
    """
    try:
        from imblearn.over_sampling import ADASYN
        
        # 수치형 컬럼만 선택
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            print("    ⚠️  수치형 특성이 없어 ADASYN을 적용할 수 없습니다.")
            return X, y
        
        adasyn = ADASYN(random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(X[numeric_cols], y)
        
        print(f"    → ADASYN 적용: {len(y_resampled)}개 샘플")
        return X_resampled, y_resampled
    
    except ImportError:
        print("    ⚠️  imbalanced-learn이 설치되지 않아 오버샘플링으로 대체")
        return apply_oversampling(X, y)


def apply_undersampling(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    언더샘플링을 적용
    
    Args:
        X: 특성 데이터프레임
        y: 타겟 변수
        
    Returns:
        (재샘플링된 특성, 재샘플링된 타겟)
    """
    try:
        from imblearn.under_sampling import RandomUnderSampler
        
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        
        print(f"    → 언더샘플링 적용: {len(y_resampled)}개 샘플")
        return X_resampled, y_resampled
    
    except ImportError:
        print("    ⚠️  imbalanced-learn이 설치되지 않아 수동 언더샘플링으로 대체")
        return manual_undersampling(X, y)


def apply_oversampling(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    오버샘플링을 적용
    
    Args:
        X: 특성 데이터프레임
        y: 타겟 변수
        
    Returns:
        (재샘플링된 특성, 재샘플링된 타겟)
    """
    try:
        from imblearn.over_sampling import RandomOverSampler
        
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        
        print(f"    → 오버샘플링 적용: {len(y_resampled)}개 샘플")
        return X_resampled, y_resampled
    
    except ImportError:
        print("    ⚠️  imbalanced-learn이 설치되지 않아 수동 오버샘플링으로 대체")
        return manual_oversampling(X, y)


def manual_undersampling(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    수동 언더샘플링
    
    Args:
        X: 특성 데이터프레임
        y: 타겟 변수
        
    Returns:
        (재샘플링된 특성, 재샘플링된 타겟)
    """
    class_counts = y.value_counts()
    min_class_count = class_counts.min()
    
    # 각 클래스에서 최소 개수만큼 샘플링
    X_resampled = pd.DataFrame()
    y_resampled = pd.Series()
    
    for class_label in y.unique():
        class_indices = y[y == class_label].index
        sampled_indices = np.random.choice(class_indices, size=min_class_count, replace=False)
        
        X_resampled = pd.concat([X_resampled, X.loc[sampled_indices]], ignore_index=True)
        y_resampled = pd.concat([y_resampled, y.loc[sampled_indices]], ignore_index=True)
    
    print(f"    → 수동 언더샘플링 적용: {len(y_resampled)}개 샘플")
    return X_resampled, y_resampled


def manual_oversampling(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    수동 오버샘플링
    
    Args:
        X: 특성 데이터프레임
        y: 타겟 변수
        
    Returns:
        (재샘플링된 특성, 재샘플링된 타겟)
    """
    class_counts = y.value_counts()
    max_class_count = class_counts.max()
    
    # 각 클래스를 최대 개수에 맞춰 오버샘플링
    X_resampled = pd.DataFrame()
    y_resampled = pd.Series()
    
    for class_label in y.unique():
        class_indices = y[y == class_label].index
        current_count = len(class_indices)
        
        if current_count < max_class_count:
            # 부족한 만큼 랜덤 샘플링 (복원)
            additional_indices = np.random.choice(class_indices, size=max_class_count - current_count, replace=True)
            all_indices = np.concatenate([class_indices, additional_indices])
        else:
            all_indices = class_indices
        
        X_resampled = pd.concat([X_resampled, X.loc[all_indices]], ignore_index=True)
        y_resampled = pd.concat([y_resampled, y.loc[all_indices]], ignore_index=True)
    
    print(f"    → 수동 오버샘플링 적용: {len(y_resampled)}개 샘플")
    return X_resampled, y_resampled


# LangGraph 노드로 사용할 수 있는 함수
imbalance_agent = RunnableLambda(handle_imbalance)