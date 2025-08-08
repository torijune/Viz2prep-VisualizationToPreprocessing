"""
범주형 변수 인코딩 전처리 에이전트
범주형 변수를 다양한 방법으로 인코딩합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def handle_categorical_encoding(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    범주형 변수를 인코딩하는 전처리 에이전트입니다.
    
    Args:
        inputs: X, Y DataFrame과 EDA 결과가 포함된 입력 딕셔너리
        
    Returns:
        전처리된 X, Y DataFrame과 코드가 포함된 딕셔너리
    """
    print("🔧 [PREPROCESSING] 범주형 변수 인코딩 시작...")
    
    X = inputs["X"]
    Y = inputs["Y"]
    eda_results = inputs.get("eda_results", {})
    
    # X와 Y의 범주형 컬럼 찾기
    X_categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    Y_categorical_columns = Y.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not X_categorical_columns and not Y_categorical_columns:
        print("✅ [PREPROCESSING] 범주형 변수가 없습니다.")
        return {
            **inputs,
            "preprocessing_code": "# 범주형 변수가 없으므로 인코딩 불필요",
            "preprocessing_summary": "범주형 변수 없음"
        }
    
    print(f"📊 [PREPROCESSING] X 범주형 변수: {X_categorical_columns}")
    print(f"📊 [PREPROCESSING] Y 범주형 변수: {Y_categorical_columns}")
    
    # 인코딩 전략 결정
    X_encoding_steps = []
    Y_encoding_steps = []
    
    # X 범주형 변수 처리
    for col in X_categorical_columns:
        unique_count = X[col].nunique()
        
        if unique_count <= 2:
            # 이진 변수는 Label Encoding
            X_encoding_steps.append({
                'column': col,
                'method': 'label',
                'reason': f'이진 변수 (고유값 {unique_count}개)'
            })
            print(f"   📊 [PREPROCESSING] X {col}: Label Encoding (이진 변수)")
        elif unique_count <= 10:
            # 10개 이하 고유값은 One-Hot Encoding
            X_encoding_steps.append({
                'column': col,
                'method': 'onehot',
                'reason': f'범주형 변수 (고유값 {unique_count}개)'
            })
            print(f"   📊 [PREPROCESSING] X {col}: One-Hot Encoding")
        else:
            # 10개 초과는 Target Encoding 또는 삭제
            X_encoding_steps.append({
                'column': col,
                'method': 'target',
                'reason': f'고차원 범주형 변수 (고유값 {unique_count}개)'
            })
            print(f"   📊 [PREPROCESSING] X {col}: Target Encoding")
    
    # Y 범주형 변수 처리 (타겟 변수는 보통 Label Encoding)
    for col in Y_categorical_columns:
        Y_encoding_steps.append({
            'column': col,
            'method': 'label',
            'reason': '타겟 변수는 Label Encoding 사용'
        })
        print(f"   📊 [PREPROCESSING] Y {col}: Label Encoding (타겟 변수)")
    
    # 전처리 코드 생성
    print("💻 [PREPROCESSING] 인코딩 코드 생성 중...")
    
    code_lines = [
        "# 범주형 변수 인코딩 (X/Y 분리)",
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.preprocessing import LabelEncoder, OneHotEncoder",
        "from sklearn.compose import ColumnTransformer",
        "",
        "def encode_categorical_variables(X, Y):",
        "    \"\"\"X와 Y의 범주형 변수를 인코딩하는 함수\"\"\"",
        "    X_processed = X.copy()",
        "    Y_processed = Y.copy()",
        ""
    ]
    
    # X 인코딩 코드 추가
    if X_encoding_steps:
        code_lines.append("    # X 범주형 변수 인코딩")
        
        # Label Encoding이 필요한 컬럼들
        X_label_cols = [step['column'] for step in X_encoding_steps if step['method'] == 'label']
        if X_label_cols:
            code_lines.append("    # Label Encoding")
            for col in X_label_cols:
                code_lines.extend([
                    f"    le_{col} = LabelEncoder()",
                    f"    X_processed['{col}'] = le_{col}.fit_transform(X_processed['{col}'])",
                    f"    print(f'X 컬럼 {col} Label Encoding 완료')"
                ])
            code_lines.append("")
        
        # One-Hot Encoding이 필요한 컬럼들
        X_onehot_cols = [step['column'] for step in X_encoding_steps if step['method'] == 'onehot']
        if X_onehot_cols:
            code_lines.extend([
                "    # One-Hot Encoding",
                "    X_onehot_columns = []",
                "    for col in X_processed.select_dtypes(include=['object', 'category']).columns:",
                "        if col in X_processed.columns:",
                "            dummies = pd.get_dummies(X_processed[col], prefix=col)",
                "            X_processed = pd.concat([X_processed, dummies], axis=1)",
                "            X_processed = X_processed.drop(columns=[col])",
                "            X_onehot_columns.extend(dummies.columns.tolist())",
                "            print(f'X 컬럼 {col} One-Hot Encoding 완료')",
                ""
            ])
        
        # Target Encoding이 필요한 컬럼들
        X_target_cols = [step['column'] for step in X_encoding_steps if step['method'] == 'target']
        if X_target_cols:
            code_lines.extend([
                "    # Target Encoding",
                "    for col in X_processed.select_dtypes(include=['object', 'category']).columns:",
                "        if col in X_processed.columns:",
                "            # 간단한 Target Encoding (평균값 사용)",
                "            target_means = X_processed.groupby(col)[Y_processed.columns[0]].mean()",
                "            X_processed[col] = X_processed[col].map(target_means)",
                "            print(f'X 컬럼 {col} Target Encoding 완료')",
                ""
            ])
    
    # Y 인코딩 코드 추가
    if Y_encoding_steps:
        code_lines.append("    # Y 범주형 변수 인코딩")
        for step in Y_encoding_steps:
            col = step['column']
            code_lines.extend([
                f"    le_Y_{col} = LabelEncoder()",
                f"    Y_processed['{col}'] = le_Y_{col}.fit_transform(Y_processed['{col}'])",
                f"    print(f'Y 컬럼 {col} Label Encoding 완료')"
            ])
        code_lines.append("")
    
    code_lines.extend([
        "    return X_processed, Y_processed",
        "",
        "# 전처리 실행",
        "X_processed, Y_processed = encode_categorical_variables(X, Y)"
    ])
    
    preprocessing_code = "\n".join(code_lines)
    
    # 전처리 실행
    print("🔄 [PREPROCESSING] 인코딩 실행 중...")
    try:
        X_processed, Y_processed = apply_basic_categorical_encoding(X, Y, X_encoding_steps, Y_encoding_steps)
        
        print(f"✅ [PREPROCESSING] 인코딩 완료")
        print(f"   📊 [PREPROCESSING] X: {X.shape} → {X_processed.shape}")
        print(f"   📊 [PREPROCESSING] Y: {Y.shape} → {Y_processed.shape}")
        
        return {
            **inputs,
            "X_processed": X_processed,
            "Y_processed": Y_processed,
            "preprocessing_code": preprocessing_code,
            "preprocessing_summary": {
                "X_categorical_encoded": len(X_encoding_steps),
                "Y_categorical_encoded": len(Y_encoding_steps),
                "X_label_encoded": len([s for s in X_encoding_steps if s['method'] == 'label']),
                "X_onehot_encoded": len([s for s in X_encoding_steps if s['method'] == 'onehot']),
                "X_target_encoded": len([s for s in X_encoding_steps if s['method'] == 'target'])
            }
        }
        
    except Exception as e:
        print(f"❌ [PREPROCESSING] 인코딩 실행 오류: {e}")
        return {
            **inputs,
            "preprocessing_code": preprocessing_code,
            "preprocessing_summary": f"인코딩 실행 오류: {str(e)}"
        }


def apply_basic_categorical_encoding(X: pd.DataFrame, Y: pd.DataFrame,
                                   X_steps: List[Dict], Y_steps: List[Dict]) -> tuple:
    """
    기본적인 범주형 변수 인코딩을 적용합니다.
    
    Args:
        X: 특성 변수 데이터프레임
        Y: 타겟 변수 데이터프레임
        X_steps: X 인코딩 단계들
        Y_steps: Y 인코딩 단계들
        
    Returns:
        tuple: (X_processed, Y_processed) 처리된 데이터프레임들
    """
    X_processed = X.copy()
    Y_processed = Y.copy()
    
    # X 처리
    for step in X_steps:
        col = step['column']
        method = step['method']
        
        if method == 'label':
            # Label Encoding
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            
        elif method == 'onehot':
            # One-Hot Encoding
            dummies = pd.get_dummies(X_processed[col], prefix=col)
            X_processed = pd.concat([X_processed, dummies], axis=1)
            X_processed = X_processed.drop(columns=[col])
            
        elif method == 'target':
            # Target Encoding (간단한 버전)
            if len(Y_processed.columns) > 0:
                target_col = Y_processed.columns[0]
                target_means = X_processed.groupby(col)[target_col].mean()
                X_processed[col] = X_processed[col].map(target_means)
                X_processed[col] = X_processed[col].fillna(target_means.mean())
    
    # Y 처리 (타겟 변수는 보통 Label Encoding)
    for step in Y_steps:
        col = step['column']
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        Y_processed[col] = le.fit_transform(Y_processed[col].astype(str))
    
    return X_processed, Y_processed


def apply_manual_categorical_encoding(X: pd.DataFrame, Y: pd.DataFrame,
                                    method: str, inputs: Dict) -> tuple:
    """
    사용자가 지정한 방법으로 범주형 변수를 인코딩합니다.
    
    Args:
        X: 특성 변수 데이터프레임
        Y: 타겟 변수 데이터프레임
        method: 인코딩 방법
        inputs: 추가 입력 정보
        
    Returns:
        tuple: (X_processed, Y_processed) 처리된 데이터프레임들
    """
    X_processed = X.copy()
    Y_processed = Y.copy()
    
    if method == "all_onehot":
        # 모든 범주형 변수를 One-Hot Encoding
        X_categorical = X_processed.select_dtypes(include=['object', 'category'])
        Y_categorical = Y_processed.select_dtypes(include=['object', 'category'])
        
        for col in X_categorical.columns:
            dummies = pd.get_dummies(X_processed[col], prefix=col)
            X_processed = pd.concat([X_processed, dummies], axis=1)
            X_processed = X_processed.drop(columns=[col])
            
        for col in Y_categorical.columns:
            dummies = pd.get_dummies(Y_processed[col], prefix=col)
            Y_processed = pd.concat([Y_processed, dummies], axis=1)
            Y_processed = Y_processed.drop(columns=[col])
            
    elif method == "all_label":
        # 모든 범주형 변수를 Label Encoding
        from sklearn.preprocessing import LabelEncoder
        
        X_categorical = X_processed.select_dtypes(include=['object', 'category'])
        Y_categorical = Y_processed.select_dtypes(include=['object', 'category'])
        
        for col in X_categorical.columns:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            
        for col in Y_categorical.columns:
            le = LabelEncoder()
            Y_processed[col] = le.fit_transform(Y_processed[col].astype(str))
    
    return X_processed, Y_processed


# LangChain Runnable으로 등록
categorical_encoding_agent = RunnableLambda(handle_categorical_encoding)