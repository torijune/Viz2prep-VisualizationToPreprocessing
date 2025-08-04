"""
데이터 로더 에이전트
CSV 파일을 읽어서 Pandas DataFrame으로 변환합니다.
"""

import pandas as pd
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda


def load_data(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    CSV 파일을 로드하여 DataFrame을 반환합니다.
    
    Args:
        inputs: 파일 경로가 포함된 입력 딕셔너리
        
    Returns:
        DataFrame이 포함된 딕셔너리
    """
    try:
        file_path = inputs["file_path"]
        df = pd.read_csv(file_path)
        print(f"데이터 로드 완료: {df.shape[0]} 행, {df.shape[1]} 열")
        return {"dataframe": df}
    except Exception as e:
        print(f"데이터 로드 오류: {e}")
        raise


# LangGraph 노드로 사용할 수 있는 함수
data_loader = RunnableLambda(load_data) 