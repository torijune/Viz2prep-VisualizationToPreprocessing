"""
특성 엔지니어링 전처리 에이전트
새로운 특성을 생성하고 기존 특성을 변환하는 다양한 방법을 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableLambda


def engineer_features(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    특성 엔지니어링을 수행하는 전처리 함수
    
    Args:
        inputs: DataFrame과 엔지니어링 방법이 포함된 입력 딕셔너리
        
    Returns:
        새로운 특성이 추가된 DataFrame이 포함된 딕셔너리
    """
    df = inputs["dataframe"].copy()
    methods = inputs.get("engineering_methods", ["auto"])  # auto, polynomial, interaction, datetime, binning
    target_column = inputs.get("target_column", None)
    
    if isinstance(methods, str):
        methods = [methods]
    
    print(f"특성 엔지니어링 시작: {methods} 방법")
    print(f"  원본 특성 수: {len(df.columns)}")
    
    engineering_info = {
        'original_features': df.columns.tolist(),
        'methods': methods,
        'new_features': []
    }
    
    for method in methods:
        if method == "auto":
            # 자동 엔지니어링: 데이터 특성에 따라 결정
            auto_methods = []
            
            # 수치형 변수가 2개 이상이면 다항식 특성 추가
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                auto_methods.append("polynomial")
            
            # 범주형 변수가 있으면 원핫 인코딩
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                auto_methods.append("categorical")
            
            # 날짜 컬럼이 있으면 시간 특성 추가
            datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if datetime_cols:
                auto_methods.append("datetime")
            
            print(f"  자동 엔지니어링: {auto_methods} 방법 적용")
            
            for auto_method in auto_methods:
                df, new_features = apply_engineering_method(df, auto_method, target_column)
                engineering_info['new_features'].extend(new_features)
        
        else:
            # 특정 방법 적용
            df, new_features = apply_engineering_method(df, method, target_column)
            engineering_info['new_features'].extend(new_features)
    
    engineering_info.update({
        'final_features': df.columns.tolist(),
        'total_new_features': len(engineering_info['new_features'])
    })
    
    print(f"특성 엔지니어링 완료: 최종 {len(df.columns)}개 특성 (추가: {len(engineering_info['new_features'])}개)")
    
    return {
        **inputs,
        "dataframe": df,
        "engineering_info": engineering_info
    }


def apply_engineering_method(df: pd.DataFrame, method: str, target_column: Optional[str]) -> tuple:
    """
    특정 엔지니어링 방법을 적용
    
    Args:
        df: 데이터프레임
        method: 적용할 방법
        target_column: 타겟 변수
        
    Returns:
        (변환된 데이터프레임, 새로 생성된 특성 리스트)
    """
    new_features = []
    
    if method == "polynomial":
        # 다항식 특성 생성
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        if len(numeric_cols) >= 2:
            # 2차 다항식 특성 생성
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    feature_name = f"{col1}_{col2}_interaction"
                    df[feature_name] = df[col1] * df[col2]
                    new_features.append(feature_name)
            
            # 제곱 특성 생성
            for col in numeric_cols:
                feature_name = f"{col}_squared"
                df[feature_name] = df[col] ** 2
                new_features.append(feature_name)
            
            print(f"    → 다항식 특성: {len(new_features)}개 생성")
    
    elif method == "interaction":
        # 상호작용 특성 생성
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        if len(numeric_cols) >= 2:
            # 모든 수치형 변수 쌍의 상호작용
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    feature_name = f"{col1}_{col2}_interaction"
                    df[feature_name] = df[col1] * df[col2]
                    new_features.append(feature_name)
            
            print(f"    → 상호작용 특성: {len(new_features)}개 생성")
    
    elif method == "datetime":
        # 날짜/시간 특성 생성
        datetime_cols = []
        
        # 날짜 컬럼 찾기
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].iloc[0])
                    datetime_cols.append(col)
                except:
                    continue
        
        for col in datetime_cols:
            try:
                df[col] = pd.to_datetime(df[col])
                
                # 시간 관련 특성 생성
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                df[f"{col}_quarter"] = df[col].dt.quarter
                
                new_features.extend([
                    f"{col}_year", f"{col}_month", f"{col}_day", 
                    f"{col}_dayofweek", f"{col}_quarter"
                ])
                
                print(f"    → 날짜 특성: {col}에서 5개 특성 생성")
            
            except Exception as e:
                print(f"    ⚠️  {col} 날짜 변환 실패: {e}")
    
    elif method == "binning":
        # 구간화 특성 생성
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        for col in numeric_cols:
            # 5개 구간으로 나누기
            feature_name = f"{col}_binned"
            df[feature_name] = pd.cut(df[col], bins=5, labels=False)
            new_features.append(feature_name)
        
        print(f"    → 구간화 특성: {len(numeric_cols)}개 생성")
    
    elif method == "categorical":
        # 범주형 변수 처리
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            if df[col].nunique() <= 10:  # 고유값이 10개 이하인 경우만
                # 빈도 기반 특성
                value_counts = df[col].value_counts()
                feature_name = f"{col}_frequency"
                df[feature_name] = df[col].map(value_counts)
                new_features.append(feature_name)
                
                # 원핫 인코딩 (상위 5개 값만)
                top_values = value_counts.head(5).index
                for value in top_values:
                    feature_name = f"{col}_{value}_onehot"
                    df[feature_name] = (df[col] == value).astype(int)
                    new_features.append(feature_name)
        
        print(f"    → 범주형 특성: {len(new_features)}개 생성")
    
    elif method == "statistical":
        # 통계적 특성 생성
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        for col in numeric_cols:
            # Z-score
            feature_name = f"{col}_zscore"
            df[feature_name] = (df[col] - df[col].mean()) / df[col].std()
            new_features.append(feature_name)
            
            # 로그 변환 (양수 값만)
            if (df[col] > 0).all():
                feature_name = f"{col}_log"
                df[feature_name] = np.log(df[col])
                new_features.append(feature_name)
        
        print(f"    → 통계적 특성: {len(new_features)}개 생성")
    
    return df, new_features


# LangGraph 노드로 사용할 수 있는 함수
feature_engineering_agent = RunnableLambda(engineer_features)