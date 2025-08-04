"""
코드 수정 에이전트
LLM이 생성한 코드에서 오류가 발생했을 때, 오류 메시지와 코드를 함께 제공하여 수정된 코드를 생성합니다.
"""

import os
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def fix_code_with_error(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    오류가 발생한 코드를 수정합니다.
    
    Args:
        inputs: 원본 코드, 오류 메시지, 데이터프레임 정보가 포함된 딕셔너리
        
    Returns:
        수정된 코드가 포함된 딕셔너리
    """
    original_code = inputs["original_code"]
    error_message = inputs["error_message"]
    dataframe_info = inputs.get("dataframe_info", "")
    
    # OpenAI 모델 초기화
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=2000
    )
    
    # 코드 수정을 위한 프롬프트 생성
    prompt = f"""
당신은 Python 코드 수정 전문가입니다. 
다음은 데이터 전처리 코드에서 발생한 오류입니다. 코드를 수정해서 오류를 해결해주세요.

=== 원본 코드 ===
{original_code}

=== 발생한 오류 ===
{error_message}

=== 데이터프레임 정보 ===
{dataframe_info}

다음 지침을 따라 코드를 수정해주세요:

1. 문법 오류 수정 (괄호, 대괄호, 따옴표 등)
2. 변수명 오타 수정
3. 라이브러리 import 문제 해결
4. 데이터프레임 컬럼 접근 오류 수정
5. 타겟 컬럼(Survived)은 절대 수정하지 마세요
6. 코드의 기능은 유지하면서 오류만 수정

수정된 코드만 반환하고 설명은 포함하지 마세요.
"""

    try:
        # LLM에게 코드 수정 요청
        response = llm.invoke([HumanMessage(content=prompt)])
        fixed_code = response.content
        
        # 코드 블록에서 실제 코드만 추출
        if "```python" in fixed_code:
            fixed_code = fixed_code.split("```python")[1].split("```")[0].strip()
        elif "```" in fixed_code:
            fixed_code = fixed_code.split("```")[1].split("```")[0].strip()
        
        print(f"코드 수정 완료: {len(fixed_code)} 문자")
        return {"fixed_code": fixed_code}
        
    except Exception as e:
        print(f"코드 수정 중 오류: {e}")
        return {"fixed_code": original_code}


# LangGraph 노드로 사용할 수 있는 함수
code_fixer_agent = RunnableLambda(fix_code_with_error) 