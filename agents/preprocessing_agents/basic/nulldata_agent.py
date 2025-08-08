"""
결측값 처리 전처리 에이전트
데이터의 결측값을 다양한 방법으로 처리합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def handle_missing_data(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    결측값을 처리하는 전처리 에이전트입니다.
    
    Args:
        inputs: X, Y DataFrame과 EDA 결과가 포함된 입력 딕셔너리
        
    Returns:
        전처리된 X, Y DataFrame과 코드가 포함된 딕셔너리
    """
    print("🔧 [PREPROCESSING] 결측값 처리 시작...")
    
    X = inputs["X"]
    Y = inputs["Y"]
    eda_results = inputs.get("eda_results", {})
    
    # X와 Y의 결측값 현황 파악
    X_missing_summary = X.isnull().sum()
    Y_missing_summary = Y.isnull().sum()
    X_total_missing = X_missing_summary.sum()
    Y_total_missing = Y_missing_summary.sum()
    
    if X_total_missing == 0 and Y_total_missing == 0:
        print("✅ [PREPROCESSING] 결측값이 없습니다.")
        return {
            **inputs,
            "preprocessing_code": "# 결측값이 없으므로 처리 불필요",
            "preprocessing_summary": "결측값 없음"
        }
    
    print(f"📊 [PREPROCESSING] X 총 {X_total_missing}개, Y 총 {Y_total_missing}개 결측값 발견")
    
    if X_total_missing > 0:
        print(f"   📋 [PREPROCESSING] X 결측값 분포:")
        for col, missing_count in X_missing_summary[X_missing_summary > 0].items():
            missing_ratio = (missing_count / len(X)) * 100
            print(f"      - {col}: {missing_count}개 ({missing_ratio:.1f}%)")
    
    if Y_total_missing > 0:
        print(f"   📋 [PREPROCESSING] Y 결측값 분포:")
        for col, missing_count in Y_missing_summary[Y_missing_summary > 0].items():
            missing_ratio = (missing_count / len(Y)) * 100
            print(f"      - {col}: {missing_count}개 ({missing_ratio:.1f}%)")
    
    # X와 Y의 수치형과 범주형 컬럼 분리
    X_numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    Y_numeric_cols = Y.select_dtypes(include=[np.number]).columns.tolist()
    Y_categorical_cols = Y.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 결측값이 있는 컬럼들
    X_cols_with_missing = X_missing_summary[X_missing_summary > 0].index.tolist()
    Y_cols_with_missing = Y_missing_summary[Y_missing_summary > 0].index.tolist()
    
    print(f"🔍 [PREPROCESSING] 결측값 처리 전략 수립 중...")
    
    # X 처리 전략 결정
    X_preprocessing_steps = []
    for col in X_cols_with_missing:
        missing_ratio = (X_missing_summary[col] / len(X)) * 100
        
        if missing_ratio > 50:
            # 50% 이상 결측값이 있는 컬럼은 삭제
            X_preprocessing_steps.append({
                'column': col,
                'action': 'drop',
                'reason': f'결측값 비율이 {missing_ratio:.1f}%로 높음'
            })
            print(f"   🗑️  [PREPROCESSING] X {col}: 삭제 (결측값 {missing_ratio:.1f}%)")
        else:
            # 50% 미만인 경우 적절한 방법으로 채우기
            if col in X_numeric_cols:
                X_preprocessing_steps.append({
                    'column': col,
                    'action': 'fill_median',
                    'reason': '수치형 변수이므로 중앙값으로 채움'
                })
                print(f"   📊 [PREPROCESSING] X {col}: 중앙값으로 채움")
            else:
                X_preprocessing_steps.append({
                    'column': col,
                    'action': 'fill_mode',
                    'reason': '범주형 변수이므로 최빈값으로 채움'
                })
                print(f"   📊 [PREPROCESSING] X {col}: 최빈값으로 채움")
    
    # Y 처리 전략 결정
    Y_preprocessing_steps = []
    for col in Y_cols_with_missing:
        missing_ratio = (Y_missing_summary[col] / len(Y)) * 100
        
        if missing_ratio > 50:
            # 50% 이상 결측값이 있는 컬럼은 삭제
            Y_preprocessing_steps.append({
                'column': col,
                'action': 'drop',
                'reason': f'결측값 비율이 {missing_ratio:.1f}%로 높음'
            })
            print(f"   🗑️  [PREPROCESSING] Y {col}: 삭제 (결측값 {missing_ratio:.1f}%)")
        else:
            # 50% 미만인 경우 적절한 방법으로 채우기
            if col in Y_numeric_cols:
                Y_preprocessing_steps.append({
                    'column': col,
                    'action': 'fill_median',
                    'reason': '수치형 변수이므로 중앙값으로 채움'
                })
                print(f"   📊 [PREPROCESSING] Y {col}: 중앙값으로 채움")
            else:
                Y_preprocessing_steps.append({
                    'column': col,
                    'action': 'fill_mode',
                    'reason': '범주형 변수이므로 최빈값으로 채움'
                })
                print(f"   📊 [PREPROCESSING] Y {col}: 최빈값으로 채움")
    
    # 전처리 코드 생성
    print("💻 [PREPROCESSING] 전처리 코드 생성 중...")
    
    code_lines = [
        "# 결측값 처리 (X/Y 분리)",
        "import pandas as pd",
        "import numpy as np",
        "",
        "def handle_missing_values(X, Y):",
        "    \"\"\"X와 Y의 결측값을 처리하는 함수\"\"\"",
        "    X_processed = X.copy()",
        "    Y_processed = Y.copy()",
        ""
    ]
    
    # X 처리 코드 추가
    if X_preprocessing_steps:
        code_lines.append("    # X 결측값 처리")
        for step in X_preprocessing_steps:
            col = step['column']
            action = step['action']
            
            if action == 'drop':
                code_lines.append(f"    X_processed = X_processed.drop(columns=['{col}'])")
                code_lines.append(f"    print(f'X 컬럼 {col} 삭제됨')")
            elif action == 'fill_median':
                code_lines.append(f"    X_processed['{col}'] = X_processed['{col}'].fillna(X_processed['{col}'].median())")
                code_lines.append(f"    print(f'X 컬럼 {col} 중앙값으로 채움')")
            elif action == 'fill_mode':
                code_lines.append(f"    mode_value = X_processed['{col}'].mode().iloc[0] if not X_processed['{col}'].mode().empty else 'unknown'")
                code_lines.append(f"    X_processed['{col}'] = X_processed['{col}'].fillna(mode_value)")
                code_lines.append(f"    print(f'X 컬럼 {col} 최빈값으로 채움')")
        code_lines.append("")
    
    # Y 처리 코드 추가
    if Y_preprocessing_steps:
        code_lines.append("    # Y 결측값 처리")
        for step in Y_preprocessing_steps:
            col = step['column']
            action = step['action']
            
            if action == 'drop':
                code_lines.append(f"    Y_processed = Y_processed.drop(columns=['{col}'])")
                code_lines.append(f"    print(f'Y 컬럼 {col} 삭제됨')")
            elif action == 'fill_median':
                code_lines.append(f"    Y_processed['{col}'] = Y_processed['{col}'].fillna(Y_processed['{col}'].median())")
                code_lines.append(f"    print(f'Y 컬럼 {col} 중앙값으로 채움')")
            elif action == 'fill_mode':
                code_lines.append(f"    mode_value = Y_processed['{col}'].mode().iloc[0] if not Y_processed['{col}'].mode().empty else 'unknown'")
                code_lines.append(f"    Y_processed['{col}'] = Y_processed['{col}'].fillna(mode_value)")
                code_lines.append(f"    print(f'Y 컬럼 {col} 최빈값으로 채움')")
        code_lines.append("")
    
    code_lines.extend([
        "    return X_processed, Y_processed",
        "",
        "# 전처리 실행",
        "X_processed, Y_processed = handle_missing_values(X, Y)"
    ])
    
    preprocessing_code = "\n".join(code_lines)
    
    # 전처리 실행
    print("🔄 [PREPROCESSING] 전처리 실행 중...")
    try:
        X_processed, Y_processed = apply_basic_missing_data_handling(X, Y, X_preprocessing_steps, Y_preprocessing_steps)
        
        print(f"✅ [PREPROCESSING] 전처리 완료")
        print(f"   📊 [PREPROCESSING] X: {X.shape} → {X_processed.shape}")
        print(f"   📊 [PREPROCESSING] Y: {Y.shape} → {Y_processed.shape}")
        
        return {
            **inputs,
            "X_processed": X_processed,
            "Y_processed": Y_processed,
            "preprocessing_code": preprocessing_code,
            "preprocessing_summary": {
                "X_missing_handled": len(X_preprocessing_steps),
                "Y_missing_handled": len(Y_preprocessing_steps),
                "X_columns_dropped": len([s for s in X_preprocessing_steps if s['action'] == 'drop']),
                "Y_columns_dropped": len([s for s in Y_preprocessing_steps if s['action'] == 'drop'])
            }
        }
        
    except Exception as e:
        print(f"❌ [PREPROCESSING] 전처리 실행 오류: {e}")
        return {
            **inputs,
            "preprocessing_code": preprocessing_code,
            "preprocessing_summary": f"전처리 실행 오류: {str(e)}"
        }


def apply_basic_missing_data_handling(X: pd.DataFrame, Y: pd.DataFrame, 
                                    X_steps: List[Dict], Y_steps: List[Dict]) -> tuple:
    """
    기본적인 결측값 처리 방법을 적용합니다.
    
    Args:
        X: 특성 변수 데이터프레임
        Y: 타겟 변수 데이터프레임
        X_steps: X 처리 단계들
        Y_steps: Y 처리 단계들
        
    Returns:
        tuple: (X_processed, Y_processed) 처리된 데이터프레임들
    """
    X_processed = X.copy()
    Y_processed = Y.copy()
    
    # X 처리
    for step in X_steps:
        col = step['column']
        action = step['action']
        
        if action == 'drop':
            X_processed = X_processed.drop(columns=[col])
        elif action == 'fill_median':
            X_processed[col] = X_processed[col].fillna(X_processed[col].median())
        elif action == 'fill_mode':
            mode_value = X_processed[col].mode().iloc[0] if not X_processed[col].mode().empty else 'unknown'
            X_processed[col] = X_processed[col].fillna(mode_value)
    
    # Y 처리
    for step in Y_steps:
        col = step['column']
        action = step['action']
        
        if action == 'drop':
            Y_processed = Y_processed.drop(columns=[col])
        elif action == 'fill_median':
            Y_processed[col] = Y_processed[col].fillna(Y_processed[col].median())
        elif action == 'fill_mode':
            mode_value = Y_processed[col].mode().iloc[0] if not Y_processed[col].mode().empty else 'unknown'
            Y_processed[col] = Y_processed[col].fillna(mode_value)
    
    return X_processed, Y_processed


def apply_manual_missing_data_handling(X: pd.DataFrame, Y: pd.DataFrame, 
                                     method: str, inputs: Dict) -> tuple:
    """
    사용자가 지정한 방법으로 결측값을 처리합니다.
    
    Args:
        X: 특성 변수 데이터프레임
        Y: 타겟 변수 데이터프레임
        method: 처리 방법
        inputs: 추가 입력 정보
        
    Returns:
        tuple: (X_processed, Y_processed) 처리된 데이터프레임들
    """
    X_processed = X.copy()
    Y_processed = Y.copy()
    
    if method == "drop_all":
        # 모든 결측값이 있는 행 삭제
        combined_df = pd.concat([X_processed, Y_processed], axis=1)
        combined_df = combined_df.dropna()
        split_index = len(X_processed.columns)
        X_processed = combined_df.iloc[:, :split_index]
        Y_processed = combined_df.iloc[:, split_index:]
        
    elif method == "fill_zero":
        # 모든 결측값을 0으로 채우기
        X_processed = X_processed.fillna(0)
        Y_processed = Y_processed.fillna(0)
        
    elif method == "fill_mean":
        # 수치형 컬럼의 결측값을 평균으로 채우기
        X_numeric = X_processed.select_dtypes(include=[np.number])
        Y_numeric = Y_processed.select_dtypes(include=[np.number])
        
        X_processed[X_numeric.columns] = X_numeric.fillna(X_numeric.mean())
        Y_processed[Y_numeric.columns] = Y_numeric.fillna(Y_numeric.mean())
        
        # 범주형 컬럼은 최빈값으로 채우기
        X_categorical = X_processed.select_dtypes(include=['object', 'category'])
        Y_categorical = Y_processed.select_dtypes(include=['object', 'category'])
        
        for col in X_categorical.columns:
            mode_value = X_processed[col].mode().iloc[0] if not X_processed[col].mode().empty else 'unknown'
            X_processed[col] = X_processed[col].fillna(mode_value)
            
        for col in Y_categorical.columns:
            mode_value = Y_processed[col].mode().iloc[0] if not Y_processed[col].mode().empty else 'unknown'
            Y_processed[col] = Y_processed[col].fillna(mode_value)
    
    return X_processed, Y_processed


# LangChain Runnable으로 등록
missing_data_agent = RunnableLambda(handle_missing_data)