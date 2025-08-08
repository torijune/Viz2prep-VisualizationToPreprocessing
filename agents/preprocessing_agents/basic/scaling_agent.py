"""
수치형 변수 스케일링 전처리 에이전트
수치형 변수를 다양한 방법으로 스케일링합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def handle_scaling(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    수치형 변수를 스케일링하는 전처리 에이전트입니다.
    
    Args:
        inputs: X, Y DataFrame과 EDA 결과가 포함된 입력 딕셔너리
        
    Returns:
        전처리된 X, Y DataFrame과 코드가 포함된 딕셔너리
    """
    print("🔧 [PREPROCESSING] 수치형 변수 스케일링 시작...")
    
    X = inputs["X"]
    Y = inputs["Y"]
    eda_results = inputs.get("eda_results", {})
    
    # X와 Y의 수치형 컬럼 찾기
    X_numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    Y_numeric_columns = Y.select_dtypes(include=[np.number]).columns.tolist()
    
    if not X_numeric_columns and not Y_numeric_columns:
        print("✅ [PREPROCESSING] 수치형 변수가 없습니다.")
        return {
            **inputs,
            "preprocessing_code": "# 수치형 변수가 없으므로 스케일링 불필요",
            "preprocessing_summary": "수치형 변수 없음"
        }
    
    print(f"📊 [PREPROCESSING] X 수치형 변수: {X_numeric_columns}")
    print(f"📊 [PREPROCESSING] Y 수치형 변수: {Y_numeric_columns}")
    
    # 스케일링 전략 결정
    X_scaling_steps = []
    Y_scaling_steps = []
    
    # X 수치형 변수 처리 (특성 변수는 스케일링 필요)
    for col in X_numeric_columns:
        # 분산과 범위를 확인하여 스케일링 방법 결정
        std_dev = X[col].std()
        value_range = X[col].max() - X[col].min()
        
        if std_dev > 1 or value_range > 10:
            # 표준편차가 크거나 범위가 넓으면 StandardScaler
            X_scaling_steps.append({
                'column': col,
                'method': 'standard',
                'reason': f'표준편차 {std_dev:.2f}, 범위 {value_range:.2f}'
            })
            print(f"   📊 [PREPROCESSING] X {col}: StandardScaler")
        else:
            # 그렇지 않으면 MinMaxScaler
            X_scaling_steps.append({
                'column': col,
                'method': 'minmax',
                'reason': f'표준편차 {std_dev:.2f}, 범위 {value_range:.2f}'
            })
            print(f"   📊 [PREPROCESSING] X {col}: MinMaxScaler")
    
    # Y 수치형 변수 처리 (타겟 변수는 보통 스케일링하지 않음, 하지만 필요시 처리)
    for col in Y_numeric_columns:
        # 타겟 변수는 회귀 문제에서만 스케일링 고려
        Y_scaling_steps.append({
            'column': col,
            'method': 'none',
            'reason': '타겟 변수는 스케일링하지 않음'
        })
        print(f"   📊 [PREPROCESSING] Y {col}: 스케일링하지 않음 (타겟 변수)")
    
    # 전처리 코드 생성
    print("💻 [PREPROCESSING] 스케일링 코드 생성 중...")
    
    code_lines = [
        "# 수치형 변수 스케일링 (X/Y 분리)",
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler",
        "",
        "def scale_numeric_variables(X, Y):",
        "    \"\"\"X와 Y의 수치형 변수를 스케일링하는 함수\"\"\"",
        "    X_processed = X.copy()",
        "    Y_processed = Y.copy()",
        ""
    ]
    
    # X 스케일링 코드 추가
    if X_scaling_steps:
        code_lines.append("    # X 수치형 변수 스케일링")
        
        # StandardScaler가 필요한 컬럼들
        X_standard_cols = [step['column'] for step in X_scaling_steps if step['method'] == 'standard']
        if X_standard_cols:
            code_lines.extend([
                "    # StandardScaler 적용",
                "    standard_scaler = StandardScaler()",
                f"    X_processed[{X_standard_cols}] = standard_scaler.fit_transform(X_processed[{X_standard_cols}])",
                f"    print(f'X 컬럼 {X_standard_cols} StandardScaler 적용 완료')",
                ""
            ])
        
        # MinMaxScaler가 필요한 컬럼들
        X_minmax_cols = [step['column'] for step in X_scaling_steps if step['method'] == 'minmax']
        if X_minmax_cols:
            code_lines.extend([
                "    # MinMaxScaler 적용",
                "    minmax_scaler = MinMaxScaler()",
                f"    X_processed[{X_minmax_cols}] = minmax_scaler.fit_transform(X_processed[{X_minmax_cols}])",
                f"    print(f'X 컬럼 {X_minmax_cols} MinMaxScaler 적용 완료')",
                ""
            ])
    
    # Y 스케일링 코드 추가 (보통 스케일링하지 않음)
    if Y_scaling_steps:
        code_lines.append("    # Y 수치형 변수 (타겟 변수는 보통 스케일링하지 않음)")
        code_lines.append("    # 필요시 아래 주석을 해제하여 스케일링 적용")
        code_lines.append("    # Y_numeric_cols = Y_processed.select_dtypes(include=[np.number]).columns")
        code_lines.append("    # if len(Y_numeric_cols) > 0:")
        code_lines.append("    #     Y_scaler = StandardScaler()")
        code_lines.append("    #     Y_processed[Y_numeric_cols] = Y_scaler.fit_transform(Y_processed[Y_numeric_cols])")
        code_lines.append("")
    
    code_lines.extend([
        "    return X_processed, Y_processed",
        "",
        "# 전처리 실행",
        "X_processed, Y_processed = scale_numeric_variables(X, Y)"
    ])
    
    preprocessing_code = "\n".join(code_lines)
    
    # 전처리 실행
    print("🔄 [PREPROCESSING] 스케일링 실행 중...")
    try:
        X_processed, Y_processed = apply_basic_scaling(X, Y, X_scaling_steps, Y_scaling_steps)
        
        print(f"✅ [PREPROCESSING] 스케일링 완료")
        print(f"   📊 [PREPROCESSING] X: {X.shape} → {X_processed.shape}")
        print(f"   📊 [PREPROCESSING] Y: {Y.shape} → {Y_processed.shape}")
        
        return {
            **inputs,
            "X_processed": X_processed,
            "Y_processed": Y_processed,
            "preprocessing_code": preprocessing_code,
            "preprocessing_summary": {
                "X_numeric_scaled": len(X_scaling_steps),
                "Y_numeric_scaled": len([s for s in Y_scaling_steps if s['method'] != 'none']),
                "X_standard_scaled": len([s for s in X_scaling_steps if s['method'] == 'standard']),
                "X_minmax_scaled": len([s for s in X_scaling_steps if s['method'] == 'minmax'])
            }
        }
        
    except Exception as e:
        print(f"❌ [PREPROCESSING] 스케일링 실행 오류: {e}")
        return {
            **inputs,
            "preprocessing_code": preprocessing_code,
            "preprocessing_summary": f"스케일링 실행 오류: {str(e)}"
        }


def apply_basic_scaling(X: pd.DataFrame, Y: pd.DataFrame,
                       X_steps: List[Dict], Y_steps: List[Dict]) -> tuple:
    """
    기본적인 수치형 변수 스케일링을 적용합니다.
    
    Args:
        X: 특성 변수 데이터프레임
        Y: 타겟 변수 데이터프레임
        X_steps: X 스케일링 단계들
        Y_steps: Y 스케일링 단계들
        
    Returns:
        tuple: (X_processed, Y_processed) 처리된 데이터프레임들
    """
    X_processed = X.copy()
    Y_processed = Y.copy()
    
    # X 처리
    for step in X_steps:
        col = step['column']
        method = step['method']
        
        if method == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_processed[col] = scaler.fit_transform(X_processed[[col]])
            
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            X_processed[col] = scaler.fit_transform(X_processed[[col]])
            
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_processed[col] = scaler.fit_transform(X_processed[[col]])
    
    # Y 처리 (보통 스케일링하지 않음)
    for step in Y_steps:
        col = step['column']
        method = step['method']
        
        if method != 'none':
            # 필요시 Y도 스케일링
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            Y_processed[col] = scaler.fit_transform(Y_processed[[col]])
    
    return X_processed, Y_processed


def apply_manual_scaling(X: pd.DataFrame, Y: pd.DataFrame,
                        method: str, inputs: Dict) -> tuple:
    """
    사용자가 지정한 방법으로 수치형 변수를 스케일링합니다.
    
    Args:
        X: 특성 변수 데이터프레임
        Y: 타겟 변수 데이터프레임
        method: 스케일링 방법
        inputs: 추가 입력 정보
        
    Returns:
        tuple: (X_processed, Y_processed) 처리된 데이터프레임들
    """
    X_processed = X.copy()
    Y_processed = Y.copy()
    
    if method == "all_standard":
        # 모든 수치형 변수를 StandardScaler로 스케일링
        from sklearn.preprocessing import StandardScaler
        
        X_numeric = X_processed.select_dtypes(include=[np.number])
        Y_numeric = Y_processed.select_dtypes(include=[np.number])
        
        if len(X_numeric.columns) > 0:
            X_scaler = StandardScaler()
            X_processed[X_numeric.columns] = X_scaler.fit_transform(X_numeric)
            
        if len(Y_numeric.columns) > 0:
            Y_scaler = StandardScaler()
            Y_processed[Y_numeric.columns] = Y_scaler.fit_transform(Y_numeric)
            
    elif method == "all_minmax":
        # 모든 수치형 변수를 MinMaxScaler로 스케일링
        from sklearn.preprocessing import MinMaxScaler
        
        X_numeric = X_processed.select_dtypes(include=[np.number])
        Y_numeric = Y_processed.select_dtypes(include=[np.number])
        
        if len(X_numeric.columns) > 0:
            X_scaler = MinMaxScaler()
            X_processed[X_numeric.columns] = X_scaler.fit_transform(X_numeric)
            
        if len(Y_numeric.columns) > 0:
            Y_scaler = MinMaxScaler()
            Y_processed[Y_numeric.columns] = Y_scaler.fit_transform(Y_numeric)
            
    elif method == "robust":
        # RobustScaler 사용 (이상치에 강함)
        from sklearn.preprocessing import RobustScaler
        
        X_numeric = X_processed.select_dtypes(include=[np.number])
        Y_numeric = Y_processed.select_dtypes(include=[np.number])
        
        if len(X_numeric.columns) > 0:
            X_scaler = RobustScaler()
            X_processed[X_numeric.columns] = X_scaler.fit_transform(X_numeric)
            
        if len(Y_numeric.columns) > 0:
            Y_scaler = RobustScaler()
            Y_processed[Y_numeric.columns] = Y_scaler.fit_transform(Y_numeric)
    
    return X_processed, Y_processed


# LangChain Runnable으로 등록
scaling_agent = RunnableLambda(handle_scaling)