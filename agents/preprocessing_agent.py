"""
전처리 에이전트
멀티모달 LLM을 사용하여 통계 정보와 시각화를 바탕으로 전처리 코드를 생성합니다.
"""

import base64
import os
from typing import Dict, Any, List
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


def encode_image_to_base64(image_path: str) -> str:
    """
    이미지를 base64로 인코딩합니다.
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        base64 인코딩된 이미지 문자열
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def create_multimodal_prompt(text_analysis: str, domain_analysis: str, plot_paths: List[str]) -> str:
    """
    멀티모달 프롬프트를 생성합니다.
    
    Args:
        text_analysis: 텍스트 분석 결과
        domain_analysis: 도메인 분석 결과
        plot_paths: 플롯 파일 경로 리스트
        
    Returns:
        멀티모달 프롬프트 문자열
    """
    prompt = f"""
당신은 데이터 전처리 및 Feature Engineering 전문가입니다. 
제공된 데이터셋 분석 결과와 도메인 지식을 바탕으로 고급 전처리 코드를 작성해주세요.

다음은 데이터셋의 상세한 텍스트 분석입니다:

{text_analysis}

다음은 도메인 분석 결과입니다:

{domain_analysis}

다음은 데이터셋의 시각화입니다:
"""
    
    # 이미지들을 base64로 인코딩하여 프롬프트에 추가
    for i, plot_path in enumerate(plot_paths):
        if os.path.exists(plot_path):
            base64_image = encode_image_to_base64(plot_path)
            prompt += f"\n[이미지 {i+1}]: data:image/png;base64,{base64_image}\n"
    
    prompt += """

위의 통계 정보와 시각화를 바탕으로 다음 작업을 수행하는 Python 코드를 작성해주세요:

1. 결측값 처리 (삭제 또는 적절한 대체)
2. 이상치 처리 (필요한 경우)
3. 범주형 변수 인코딩
4. 특성 스케일링 (필요한 경우)
5. 전처리된 DataFrame 반환

코드는 다음과 같은 형식으로 작성해주세요:
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    # 여기에 전처리 코드를 작성
    # 주의: 타겟 컬럼(Survived)은 절대 수정하지 마세요!
    return preprocessed_df

# 전처리 실행
preprocessed_df = preprocess_data(df)
```

코드만 작성하고 설명은 포함하지 마세요.
"""
    
    return prompt


def execute_preprocessing_code(code: str, df: pd.DataFrame, code_fixer_agent=None) -> pd.DataFrame:
    """
    생성된 전처리 코드를 안전하게 실행합니다.
    타겟 컬럼(Survived 등)은 전처리에서 제외하고, 전처리 후 다시 붙입니다.
    오류가 발생하면 코드 수정 에이전트를 사용하여 코드를 수정합니다.
    """
    max_retries = 3
    current_code = code
    
    for attempt in range(max_retries):
        try:
            # 타겟 컬럼 보호
            target_col = None
            for col in ['Survived', 'target', 'label', 'class', 'y']:
                if col in df.columns:
                    target_col = col
                    break
            y = None
            if target_col:
                y = df[target_col].copy()
                X = df.drop(columns=[target_col])
            else:
                X = df.copy()

            # 필요한 import 문들을 코드 앞에 자동 추가
            required_imports = []
            
            # numpy 관련 함수들 확인
            if any(func in current_code for func in ['np.', 'np.where', 'np.log', 'np.log1p', 'np.exp', 'np.sqrt']):
                if 'import numpy' not in current_code and 'import np' not in current_code:
                    required_imports.append("import numpy as np")
            
            # pandas 관련 함수들 확인
            if any(func in current_code for func in ['pd.', 'fillna', 'groupby', 'apply', 'drop']):
                if 'import pandas' not in current_code and 'import pd' not in current_code:
                    required_imports.append("import pandas as pd")
            
            # sklearn 관련 함수들 확인
            if any(func in current_code for func in ['StandardScaler', 'LabelEncoder', 'MinMaxScaler', 'SimpleImputer']):
                if 'from sklearn' not in current_code:
                    required_imports.append("from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler")
                    required_imports.append("from sklearn.impute import SimpleImputer")
            
            # 필요한 import 문들을 코드 앞에 추가
            if required_imports:
                current_code = "\n".join(required_imports) + "\n\n" + current_code
                
            # 타겟 컬럼 보호를 위한 코드 수정
            if target_col:
                # 타겟 컬럼이 포함된 컬럼 선택 코드를 수정
                current_code = current_code.replace(f"['{target_col}']", "[]")
                current_code = current_code.replace(f"'{target_col}'", "''")
                current_code = current_code.replace(f'"{target_col}"', '""')
                # df[['PassengerId', 'Survived', ...]] 형태의 코드 수정
                import re
                pattern = r'df\[\[([^\]]*)\]\].*?\]'
                def replace_target_col(match):
                    cols_str = match.group(1)
                    cols = [col.strip().strip("'\"") for col in cols_str.split(',')]
                    cols = [col for col in cols if col != target_col]
                    return f'df[{[repr(col) for col in cols]}]'
                
                current_code = re.sub(pattern, replace_target_col, current_code)

            # 안전한 실행을 위한 로컬 네임스페이스
            local_namespace = {
                'df': X.copy(),
                'pd': pd,
                'np': np,
                'StandardScaler': StandardScaler,
                'LabelEncoder': LabelEncoder,
                'SimpleImputer': SimpleImputer,
                'sklearn': __import__('sklearn'),
                'sklearn.preprocessing': __import__('sklearn.preprocessing'),
                'sklearn.impute': __import__('sklearn.impute'),
                'sklearn.model_selection': __import__('sklearn.model_selection'),
                'sklearn.ensemble': __import__('sklearn.ensemble')
            }
            
            # 전역 네임스페이스도 동일하게 설정
            global_namespace = {
                'pd': pd,
                'np': np,
                'StandardScaler': StandardScaler,
                'LabelEncoder': LabelEncoder,
                'SimpleImputer': SimpleImputer,
                'sklearn': __import__('sklearn'),
                'sklearn.preprocessing': __import__('sklearn.preprocessing'),
                'sklearn.impute': __import__('sklearn.impute'),
                'sklearn.model_selection': __import__('sklearn.model_selection'),
                'sklearn.ensemble': __import__('sklearn.ensemble')
            }
            
            # 코드에서 함수 정의 부분 추출
            if 'def preprocess_data(df):' in current_code:
                exec(current_code, global_namespace, local_namespace)
                if 'preprocess_data' in local_namespace:
                    preprocessed_df = local_namespace['preprocess_data'](X)
                else:
                    raise ValueError("preprocess_data 함수를 찾을 수 없습니다.")
            else:
                exec(current_code, global_namespace, local_namespace)
                if 'preprocessed_df' in local_namespace:
                    preprocessed_df = local_namespace['preprocessed_df']
                else:
                    raise ValueError("preprocessed_df 변수를 찾을 수 없습니다.")
            
            # 전처리 후 타겟 컬럼 다시 붙이기 (인덱스 맞춤)
            if target_col and isinstance(preprocessed_df, pd.DataFrame):
                # 인덱스가 맞지 않을 경우를 대비해 인덱스 리셋
                preprocessed_df = preprocessed_df.reset_index(drop=True)
                y = y.reset_index(drop=True)
                preprocessed_df[target_col] = y.values
            
            return preprocessed_df
            
        except Exception as e:
            error_message = str(e)
            print(f"전처리 코드 실행 오류 (시도 {attempt + 1}/{max_retries}): {error_message}")
            
            # 코드 수정 에이전트가 있고 마지막 시도가 아니라면 코드 수정 시도
            if code_fixer_agent and attempt < max_retries - 1:
                try:
                    print("코드 수정 에이전트를 사용하여 코드를 수정합니다...")
                    dataframe_info = f"컬럼: {list(df.columns)}\n데이터 타입: {df.dtypes.to_dict()}"
                    
                    fix_result = code_fixer_agent.invoke({
                        "original_code": current_code,
                        "error_message": error_message,
                        "dataframe_info": dataframe_info
                    })
                    
                    current_code = fix_result.get("fixed_code", current_code)
                    print("코드 수정 완료, 다시 실행합니다...")
                    
                except Exception as fix_error:
                    print(f"코드 수정 중 오류: {fix_error}")
                    break
            else:
                break
    
    # 모든 시도가 실패한 경우 기본 전처리 사용
    print("모든 코드 수정 시도 실패, 기본 전처리를 사용합니다.")
    return perform_basic_preprocessing(df)


def perform_basic_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    기본 전처리를 수행합니다 (LLM 코드 실행 실패 시 사용).
    
    Args:
        df: 원본 DataFrame
        
    Returns:
        전처리된 DataFrame
    """
    print("기본 전처리 수행 중...")
    
    # numpy import 확인
    import numpy as np
    
    df_processed = df.copy()
    
    # 결측값 처리
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    
    # 수치형 컬럼 결측값을 중앙값으로 대체
    if len(numeric_cols) > 0:
        imputer = SimpleImputer(strategy='median')
        df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
    
    # 범주형 컬럼 결측값을 최빈값으로 대체
    if len(categorical_cols) > 0:
        imputer = SimpleImputer(strategy='most_frequent')
        df_processed[categorical_cols] = imputer.fit_transform(df_processed[categorical_cols])
    
    # 범주형 변수 인코딩
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    # Feature Engineering
    if 'SibSp' in df_processed.columns and 'Parch' in df_processed.columns:
        df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
        df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)
    
    if 'Fare' in df_processed.columns and 'FamilySize' in df_processed.columns:
        df_processed['FarePerPerson'] = df_processed['Fare'] / df_processed['FamilySize']
    
    if 'Age' in df_processed.columns and 'Pclass' in df_processed.columns:
        df_processed['Age*Class'] = df_processed['Age'] * df_processed['Pclass']
    
    if 'Name' in df_processed.columns:
        # Name 컬럼이 문자열인지 확인하고 Title 추출
        if df_processed['Name'].dtype == 'object':
            df_processed['Title'] = df_processed['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
            # Title이 없는 경우 'Unknown'으로 대체
            df_processed['Title'] = df_processed['Title'].fillna('Unknown')
        else:
            # Name이 문자열이 아닌 경우 간단한 Title 생성
            df_processed['Title'] = 'Unknown'
    
    # 수치형 변수 스케일링 (새로 추가된 컬럼들 포함)
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    
    return df_processed


def generate_preprocessing_code(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    멀티모달 LLM을 사용하여 고급 전처리 및 Feature Engineering 코드를 생성합니다.
    
    Args:
        inputs: DataFrame, 텍스트 분석, 도메인 분석, 플롯 경로가 포함된 입력 딕셔너리
        
    Returns:
        전처리 코드와 전처리된 DataFrame이 포함된 딕셔너리
    """
    df = inputs["dataframe"]
    text_analysis = inputs["text_analysis"]
    domain_analysis = inputs.get("domain_analysis", "")
    plot_paths = inputs["plot_paths"]
    
    # OpenAI 멀티모달 모델 초기화
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=2000
    )
    
    # 멀티모달 프롬프트 생성
    prompt = create_multimodal_prompt(text_analysis, domain_analysis, plot_paths)
    
    # 멀티모달 메시지 생성
    messages = []
    
    # 통계 텍스트와 이미지들을 하나의 메시지로 결합
    content = []
    
    # 텍스트 분석 및 도메인 분석 추가
    content.append({
        "type": "text", 
        "text": f"""당신은 데이터 전처리 및 Feature Engineering 전문가입니다. 
제공된 데이터셋 분석 결과와 도메인 지식을 바탕으로 고급 전처리 코드를 작성해주세요.

다음은 데이터셋의 상세한 텍스트 분석입니다:

{text_analysis}

다음은 도메인 분석 결과입니다:

{domain_analysis}

다음은 데이터셋의 시각화입니다:"""
    })
    
    # 이미지들 추가
    for i, plot_path in enumerate(plot_paths):
        if os.path.exists(plot_path):
            with open(plot_path, "rb") as image_file:
                image_content = image_file.read()
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(image_content).decode('utf-8')}"
                    }
                })
    
    # 전처리 요청 텍스트 추가
    content.append({
        "type": "text",
        "text": """

위의 분석 결과와 도메인 지식을 바탕으로 다음 고급 작업을 수행하는 Python 코드를 작성해주세요:

1. **고급 결측값 처리**: 도메인 지식을 활용한 스마트한 결측치 대체
2. **이상치 탐지 및 처리**: 비즈니스 로직에 기반한 이상치 정의와 처리
3. **Feature Engineering**: 
   - 기존 변수들을 조합한 새로운 특성 생성
   - 비즈니스 로직에 기반한 파생 변수 생성
   - 도메인 전문가가 고려할 수 있는 숨겨진 패턴 활용
4. **범주형 변수 인코딩**: 도메인 특성에 맞는 인코딩 방법 선택
5. **스케일링 및 정규화**: 비즈니스 컨텍스트에 적합한 방법 적용
6. **특성 선택**: 도메인 중요도에 따른 특성 우선순위 적용

코드는 다음과 같은 형식으로 작성해주세요:
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    # 여기에 전처리 코드를 작성
    # 주의: 타겟 컬럼(Survived)은 절대 수정하지 마세요!
    # 주의: DataFrame 인덱스를 변경하지 마세요 (reset_index 사용 금지)
    # 주의: 새로운 컬럼 추가 시 df['new_col'] = values 형태로 추가하세요
    
    # 예시:
    # df['new_feature'] = df['col1'] + df['col2']  # 올바른 방법
    # df = df.assign(new_feature=df['col1'] + df['col2'])  # 올바른 방법
    
    return df

# 전처리 실행
preprocessed_df = preprocess_data(df)
```

코드만 작성하고 설명은 포함하지 마세요."""
    })
    
    messages.append(HumanMessage(content=content))
    
    try:
        # LLM 호출
        response = llm.invoke(messages)
        generated_code = response.content
        
        # 코드에서 Python 코드 블록 추출
        if "```python" in generated_code:
            code_start = generated_code.find("```python") + 9
            code_end = generated_code.find("```", code_start)
            if code_end != -1:
                generated_code = generated_code[code_start:code_end].strip()
        elif "```" in generated_code:
            code_start = generated_code.find("```") + 3
            code_end = generated_code.find("```", code_start)
            if code_end != -1:
                generated_code = generated_code[code_start:code_end].strip()
        
        print("LLM이 생성한 전처리 코드:")
        print(generated_code)
        
        # 코드 수정 에이전트 import
        from agents.code_fixer_agent import code_fixer_agent
        
        # 코드 실행 (코드 수정 에이전트 포함)
        preprocessed_df = execute_preprocessing_code(generated_code, df, code_fixer_agent)
        
        return {
            "dataframe": df,
            "preprocessed_dataframe": preprocessed_df,
            "preprocessing_code": generated_code
        }
        
    except Exception as e:
        print(f"LLM 호출 오류: {e}")
        # 기본 전처리 수행
        preprocessed_df = perform_basic_preprocessing(df)
        
        return {
            "dataframe": df,
            "preprocessed_dataframe": preprocessed_df,
            "preprocessing_code": "# 기본 전처리 수행됨 (LLM 오류)"
        }


# LangGraph 노드로 사용할 수 있는 함수
preprocessing_agent = RunnableLambda(generate_preprocessing_code) 