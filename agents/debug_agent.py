"""
Code Debug Agent
전처리 코드 실행 시 발생하는 오류를 자동으로 디버깅하고 수정합니다.
"""

import traceback
import re
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import pandas as pd
import numpy as np


def debug_preprocessing_code(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    전처리 코드 실행 시 발생하는 오류를 디버깅하고 수정합니다.
    
    Args:
        inputs: X, Y, 전처리 코드, 오류 정보가 포함된 입력 딕셔너리
        
    Returns:
        수정된 전처리 코드와 디버깅 정보가 포함된 딕셔너리
    """
    print("🐛 [DEBUG] Code Debug Agent 시작...")
    
    X = inputs.get("X")  # 특성 변수들
    Y = inputs.get("Y")  # 타겟 변수
    preprocessing_code = inputs.get("preprocessing_code", "")
    error_message = inputs.get("error_message", "")
    execution_result = inputs.get("execution_result", {})
    
    # X, Y 필수 입력 검증
    if X is None or Y is None:
        error_msg = "X와 Y 데이터프레임이 모두 필요합니다."
        print(f"❌ [DEBUG] {error_msg}")
        return {
            **inputs,
            "debug_status": "error",
            "debug_message": error_msg,
            "fixed_preprocessing_code": "",
            "debug_analysis": "X와 Y 데이터프레임이 누락되었습니다."
        }
    
    if not error_message:
        print("✅ [DEBUG] 오류가 없으므로 디버깅 불필요")
        return {
            **inputs,
            "debug_status": "no_error",
            "debug_message": "오류가 없습니다.",
            "fixed_preprocessing_code": "",
            "debug_analysis": "오류가 없어 수정이 필요하지 않습니다."
        }
    
    print(f"🔍 [DEBUG] 오류 분석 중...")
    print(f"   📝 [DEBUG] 오류 메시지: {error_message}")
    print(f"   📊 [DEBUG] X shape: {X.shape}, Y shape: {Y.shape}")
    
    # 오류 분석 및 수정 (X/Y 분리 방식만 사용)
    debug_result = analyze_and_fix_error(X, Y, preprocessing_code, error_message)
    
    return {
        **inputs,
        "debug_status": debug_result["status"],
        "debug_message": debug_result["message"],
        "fixed_preprocessing_code": debug_result["fixed_code"],
        "debug_analysis": debug_result["analysis"]
    }


def analyze_and_fix_error(X: pd.DataFrame, Y: pd.DataFrame, code: str, error_msg: str) -> Dict[str, Any]:
    """
    X와 Y가 분리된 상태에서 오류를 분석하고 수정된 코드를 생성합니다.
    
    Args:
        X: 특성 변수 데이터프레임
        Y: 타겟 변수 데이터프레임
        code: 원본 전처리 코드
        error_msg: 오류 메시지
        
    Returns:
        디버깅 결과 딕셔너리
    """
    # 일반적인 오류 패턴 분석
    error_patterns = {
        r"target_variable.*존재하지 않습니다": "target_variable_missing",
        r"컬럼.*존재하지 않습니다": "column_missing",
        r"인덱스.*범위를 벗어났습니다": "index_error",
        r"데이터 타입.*오류": "data_type_error",
        r"메모리.*부족": "memory_error",
        r"0으로 나누기": "division_by_zero",
        r"NaN.*처리": "nan_handling",
        r"인코딩.*오류": "encoding_error"
    }
    
    error_type = "unknown"
    for pattern, error_type_name in error_patterns.items():
        if re.search(pattern, error_msg, re.IGNORECASE):
            error_type = error_type_name
            break
    
    print(f"🔍 [DEBUG] 오류 타입: {error_type}")
    print(f"   📊 [DEBUG] X shape: {X.shape}, Y shape: {Y.shape}")
    
    # 오류 타입별 수정 전략
    if error_type == "target_variable_missing":
        return fix_target_variable_error(X, Y, code, error_msg)
    elif error_type == "column_missing":
        return fix_column_missing_error(X, Y, code, error_msg)
    elif error_type == "data_type_error":
        return fix_data_type_error(X, Y, code, error_msg)
    elif error_type == "nan_handling":
        return fix_nan_handling_error(X, Y, code, error_msg)
    else:
        return fix_generic_error(X, Y, code, error_msg)


def fix_target_variable_error(X: pd.DataFrame, Y: pd.DataFrame, code: str, error_msg: str) -> Dict[str, Any]:
    """
    target_variable 관련 오류를 수정합니다.
    """
    print("🔧 [DEBUG] target_variable 오류 수정 중...")
    
    # X와 Y의 컬럼 정보
    X_columns = X.columns.tolist()
    Y_columns = Y.columns.tolist()
    
    print(f"   📋 [DEBUG] X 컬럼: {X_columns}")
    print(f"   📋 [DEBUG] Y 컬럼: {Y_columns}")
    
    # target_variable을 Y에서 찾거나 생성하는 코드 추가
    fixed_code = f"""# target_variable 오류 수정
import pandas as pd
import numpy as np

def preprocess_data(X, Y):
    \"\"\"전처리 함수 - target_variable 오류 수정\"\"\"
    X_processed = X.copy()
    Y_processed = Y.copy()
    
    # X와 Y 컬럼 정보
    X_columns = {X_columns}
    Y_columns = {Y_columns}
    print(f"X 컬럼: {{X_columns}}")
    print(f"Y 컬럼: {{Y_columns}}")
    
    # target_variable이 Y에 없는 경우 처리
    if 'target_variable' not in Y_processed.columns:
        print("target_variable 컬럼이 Y에 없습니다.")
        
        if Y_columns:
            # Y의 첫 번째 컬럼을 target_variable로 사용
            target_col = Y_columns[0]
            print(f"'{target_col}'를 target_variable로 사용합니다.")
            Y_processed['target_variable'] = Y_processed[target_col]
        else:
            # Y가 비어있는 경우 X에서 수치형 컬럼을 찾아서 생성
            numeric_columns = X_processed.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_columns:
                target_col = numeric_columns[0]
                print(f"X의 '{target_col}'를 target_variable로 사용합니다.")
                Y_processed['target_variable'] = X_processed[target_col]
                X_processed = X_processed.drop(columns=[target_col])
            else:
                print("적절한 target_variable을 찾을 수 없어 더미 값을 생성합니다.")
                Y_processed['target_variable'] = 0
    
    # 원본 전처리 코드 실행
{code}
    
    return X_processed, Y_processed

# 전처리 실행
X_processed, Y_processed = preprocess_data(X, Y)
"""
    
    return {
        "status": "fixed",
        "message": "target_variable 오류를 수정했습니다. X와 Y를 분리하여 처리하고 적절한 target_variable을 설정합니다.",
        "fixed_code": fixed_code,
        "analysis": f"X 컬럼: {X_columns}, Y 컬럼: {Y_columns}"
    }


def fix_column_missing_error(X: pd.DataFrame, Y: pd.DataFrame, code: str, error_msg: str) -> Dict[str, Any]:
    """
    컬럼 누락 오류를 수정합니다.
    """
    print("🔧 [DEBUG] 컬럼 누락 오류 수정 중...")
    
    # 누락된 컬럼 찾기
    missing_column = extract_missing_column(error_msg)
    X_columns = X.columns.tolist()
    Y_columns = Y.columns.tolist()
    
    print(f"   📋 [DEBUG] 누락된 컬럼: {missing_column}")
    print(f"   📋 [DEBUG] X 컬럼: {X_columns}")
    print(f"   📋 [DEBUG] Y 컬럼: {Y_columns}")
    
    # 유사한 컬럼 찾기 (X와 Y 모두에서)
    similar_column = find_similar_column(missing_column, X_columns + Y_columns)
    
    fixed_code = f"""# 컬럼 누락 오류 수정
import pandas as pd
import numpy as np

def preprocess_data(X, Y):
    \"\"\"전처리 함수 - 컬럼 누락 오류 수정\"\"\"
    X_processed = X.copy()
    Y_processed = Y.copy()
    
    # 컬럼 정보
    X_columns = {X_columns}
    Y_columns = {Y_columns}
    print(f"X 컬럼: {{X_columns}}")
    print(f"Y 컬럼: {{Y_columns}}")
    
    # 누락된 컬럼 처리
    missing_column = "{missing_column}"
    all_columns = X_columns + Y_columns
    
    if missing_column not in all_columns:
        print(f"'{missing_column}' 컬럼이 X 또는 Y에 없습니다.")
        
        # 유사한 컬럼 찾기
        similar_column = "{similar_column}"
        if similar_column:
            if similar_column in X_columns:
                print(f"X의 '{similar_column}' 컬럼을 사용합니다.")
                if missing_column not in X_processed.columns:
                    X_processed[missing_column] = X_processed[similar_column]
            elif similar_column in Y_columns:
                print(f"Y의 '{similar_column}' 컬럼을 사용합니다.")
                if missing_column not in Y_processed.columns:
                    Y_processed[missing_column] = Y_processed[similar_column]
        else:
            # 기본값으로 컬럼 생성
            print(f"'{missing_column}' 컬럼을 기본값으로 생성합니다.")
            if X_processed.select_dtypes(include=[np.number]).columns.any():
                X_processed[missing_column] = 0
            else:
                X_processed[missing_column] = "unknown"
    
    # 원본 전처리 코드 실행
{code}
    
    return X_processed, Y_processed

# 전처리 실행
X_processed, Y_processed = preprocess_data(X, Y)
"""
    
    return {
        "status": "fixed",
        "message": f"컬럼 누락 오류를 수정했습니다. '{missing_column}' 컬럼을 X/Y 분리 상태에서 적절히 처리합니다.",
        "fixed_code": fixed_code,
        "analysis": f"누락된 컬럼: {missing_column}, 유사한 컬럼: {similar_column}"
    }


def fix_data_type_error(X: pd.DataFrame, Y: pd.DataFrame, code: str, error_msg: str) -> Dict[str, Any]:
    """
    데이터 타입 오류를 수정합니다.
    """
    print("🔧 [DEBUG] 데이터 타입 오류 수정 중...")
    
    # 데이터 타입 정보 수집
    X_dtypes = X.dtypes.to_dict()
    Y_dtypes = Y.dtypes.to_dict()
    
    fixed_code = f"""# 데이터 타입 오류 수정
import pandas as pd
import numpy as np

def preprocess_data(X, Y):
    \"\"\"전처리 함수 - 데이터 타입 오류 수정\"\"\"
    X_processed = X.copy()
    Y_processed = Y.copy()
    
    # 데이터 타입 정보 출력
    print("X 데이터 타입 정보:")
    for col, dtype in X_processed.dtypes.items():
        print(f"  {{col}}: {{dtype}}")
    
    print("Y 데이터 타입 정보:")
    for col, dtype in Y_processed.dtypes.items():
        print(f"  {{col}}: {{dtype}}")
    
    # 데이터 타입 변환 시도
    X_numeric_columns = X_processed.select_dtypes(include=[np.number]).columns.tolist()
    X_categorical_columns = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    
    Y_numeric_columns = Y_processed.select_dtypes(include=[np.number]).columns.tolist()
    Y_categorical_columns = Y_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"X 수치형 컬럼: {{X_numeric_columns}}")
    print(f"X 범주형 컬럼: {{X_categorical_columns}}")
    print(f"Y 수치형 컬럼: {{Y_numeric_columns}}")
    print(f"Y 범주형 컬럼: {{Y_categorical_columns}}")
    
    # 안전한 데이터 타입 변환 (X)
    for col in X_processed.columns:
        if col in X_categorical_columns:
            X_processed[col] = X_processed[col].astype(str)
        elif col in X_numeric_columns:
            try:
                X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
            except:
                print(f"X '{col}' 컬럼 변환 실패, 원본 유지")
    
    # 안전한 데이터 타입 변환 (Y)
    for col in Y_processed.columns:
        if col in Y_categorical_columns:
            Y_processed[col] = Y_processed[col].astype(str)
        elif col in Y_numeric_columns:
            try:
                Y_processed[col] = pd.to_numeric(Y_processed[col], errors='coerce')
            except:
                print(f"Y '{col}' 컬럼 변환 실패, 원본 유지")
    
    # 원본 전처리 코드 실행
{code}
    
    return X_processed, Y_processed

# 전처리 실행
X_processed, Y_processed = preprocess_data(X, Y)
"""
    
    return {
        "status": "fixed",
        "message": "데이터 타입 오류를 수정했습니다. X와 Y를 분리하여 안전한 데이터 타입 변환을 적용합니다.",
        "fixed_code": fixed_code,
        "analysis": f"X 데이터 타입: {X_dtypes}, Y 데이터 타입: {Y_dtypes}"
    }


def fix_nan_handling_error(X: pd.DataFrame, Y: pd.DataFrame, code: str, error_msg: str) -> Dict[str, Any]:
    """
    NaN 처리 오류를 수정합니다.
    """
    print("🔧 [DEBUG] NaN 처리 오류 수정 중...")
    
    # NaN 현황 확인
    X_nan_summary = X.isnull().sum()
    Y_nan_summary = Y.isnull().sum()
    total_X_nan = X_nan_summary.sum()
    total_Y_nan = Y_nan_summary.sum()
    
    fixed_code = f"""# NaN 처리 오류 수정
import pandas as pd
import numpy as np

def preprocess_data(X, Y):
    \"\"\"전처리 함수 - NaN 처리 오류 수정\"\"\"
    X_processed = X.copy()
    Y_processed = Y.copy()
    
    # NaN 현황 확인
    X_nan_summary = X_processed.isnull().sum()
    Y_nan_summary = Y_processed.isnull().sum()
    total_X_nan = X_nan_summary.sum()
    total_Y_nan = Y_nan_summary.sum()
    
    print(f"X 총 NaN 개수: {{total_X_nan}}")
    print(f"Y 총 NaN 개수: {{total_Y_nan}}")
    
    # X의 NaN 처리
    if total_X_nan > 0:
        print("X의 NaN 처리 중...")
        for col in X_processed.columns:
            if X_processed[col].isnull().sum() > 0:
                if X_processed[col].dtype in ['object', 'category']:
                    mode_value = X_processed[col].mode().iloc[0] if not X_processed[col].mode().empty else "unknown"
                    X_processed[col] = X_processed[col].fillna(mode_value)
                    print(f"X '{col}': 최빈값으로 채움")
                else:
                    median_value = X_processed[col].median()
                    X_processed[col] = X_processed[col].fillna(median_value)
                    print(f"X '{col}': 중앙값으로 채움")
    
    # Y의 NaN 처리
    if total_Y_nan > 0:
        print("Y의 NaN 처리 중...")
        for col in Y_processed.columns:
            if Y_processed[col].isnull().sum() > 0:
                if Y_processed[col].dtype in ['object', 'category']:
                    mode_value = Y_processed[col].mode().iloc[0] if not Y_processed[col].mode().empty else "unknown"
                    Y_processed[col] = Y_processed[col].fillna(mode_value)
                    print(f"Y '{col}': 최빈값으로 채움")
                else:
                    median_value = Y_processed[col].median()
                    Y_processed[col] = Y_processed[col].fillna(median_value)
                    print(f"Y '{col}': 중앙값으로 채움")
    
    # 원본 전처리 코드 실행
{code}
    
    return X_processed, Y_processed

# 전처리 실행
X_processed, Y_processed = preprocess_data(X, Y)
"""
    
    return {
        "status": "fixed",
        "message": "NaN 처리 오류를 수정했습니다. X와 Y를 분리하여 안전한 NaN 처리 방법을 적용합니다.",
        "fixed_code": fixed_code,
        "analysis": f"X 총 NaN 개수: {total_X_nan}, Y 총 NaN 개수: {total_Y_nan}"
    }


def fix_generic_error(X: pd.DataFrame, Y: pd.DataFrame, code: str, error_msg: str) -> Dict[str, Any]:
    """
    일반적인 오류를 수정합니다.
    """
    print("🔧 [DEBUG] 일반적인 오류 수정 중...")
    
    # 안전한 전처리 코드 생성
    fixed_code = f"""# 일반적인 오류 수정
import pandas as pd
import numpy as np

def preprocess_data(X, Y):
    \"\"\"전처리 함수 - 안전한 실행\"\"\"
    try:
        X_processed = X.copy()
        Y_processed = Y.copy()
        
        # 기본 정보 출력
        print(f"X 데이터프레임 크기: {{X_processed.shape}}")
        print(f"Y 데이터프레임 크기: {{Y_processed.shape}}")
        print(f"X 컬럼 목록: {{list(X_processed.columns)}}")
        print(f"Y 컬럼 목록: {{list(Y_processed.columns)}}")
        
        # 안전한 전처리 실행
{code}
        
        return X_processed, Y_processed
        
    except Exception as e:
        print(f"오류 발생: {{e}}")
        print("기본 전처리만 수행합니다.")
        
        # 기본 전처리
        X_processed = X.copy()
        Y_processed = Y.copy()
        
        # X의 결측값 처리
        for col in X_processed.columns:
            if X_processed[col].isnull().sum() > 0:
                if X_processed[col].dtype in ['object', 'category']:
                    X_processed[col] = X_processed[col].fillna("unknown")
                else:
                    X_processed[col] = X_processed[col].fillna(X_processed[col].median())
        
        # Y의 결측값 처리
        for col in Y_processed.columns:
            if Y_processed[col].isnull().sum() > 0:
                if Y_processed[col].dtype in ['object', 'category']:
                    Y_processed[col] = Y_processed[col].fillna("unknown")
                else:
                    Y_processed[col] = Y_processed[col].fillna(Y_processed[col].median())
        
        return X_processed, Y_processed

# 전처리 실행
X_processed, Y_processed = preprocess_data(X, Y)
"""
    
    return {
        "status": "fixed",
        "message": "일반적인 오류를 수정했습니다. X와 Y를 분리하여 안전한 전처리를 수행합니다.",
        "fixed_code": fixed_code,
        "analysis": f"오류 메시지: {error_msg}"
    }


def extract_missing_column(error_msg: str) -> str:
    """오류 메시지에서 누락된 컬럼명을 추출합니다."""
    patterns = [
        r"컬럼\s*['\"]([^'\"]+)['\"]\s*존재하지 않습니다",
        r"column\s*['\"]([^'\"]+)['\"]\s*not found",
        r"([a-zA-Z_][a-zA-Z0-9_]*)\s*컬럼이 존재하지 않습니다"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, error_msg, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return "unknown_column"


def find_similar_column(target: str, available_columns: List[str]) -> Optional[str]:
    """유사한 컬럼을 찾습니다."""
    if not available_columns:
        return None
    
    # 정확한 매치
    if target in available_columns:
        return target
    
    # 부분 매치
    for col in available_columns:
        if target.lower() in col.lower() or col.lower() in target.lower():
            return col
    
    # 첫 번째 컬럼 반환
    return available_columns[0] if available_columns else None


def debug_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Code Debug Agent 노드 함수
    """
    print("🐛 [DEBUG] Code Debug Agent 노드 실행...")
    
    try:
        result = debug_preprocessing_code(state)
        return result
    except Exception as e:
        print(f"❌ [DEBUG] Code Debug Agent 오류: {e}")
        return {
            **state,
            "debug_status": "error",
            "debug_message": f"디버깅 중 오류 발생: {str(e)}",
            "fixed_preprocessing_code": state.get("preprocessing_code", "")
        } 