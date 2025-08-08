#!/usr/bin/env python3
"""
전문 플래너 노드들
각 도메인별로 전문화된 전처리 계획을 수립하는 노드들
"""

import os
import sys
import json
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflow_state import WorkflowState


def load_preprocessing_techniques():
    """
    KB에서 전처리 방법들을 로드합니다.
    """
    kb_path = os.path.join(os.path.dirname(__file__), "..", "KB_rag_agents", "knowledge_base", "preprocessing_codes.json")
    
    try:
        with open(kb_path, 'r', encoding='utf-8') as f:
            kb_data = json.load(f)
        
        # 카테고리별로 전처리 방법들을 정리
        techniques_by_category = {}
        for category, data in kb_data.items():
            techniques = [tech['name'] for tech in data['techniques']]
            techniques_by_category[category] = {
                'description': data['description'],
                'techniques': techniques
            }
        
        return techniques_by_category
    except Exception as e:
        print(f"⚠️ [PLAN] KB 로드 실패: {e}")
        return {}


def get_available_techniques(category: str) -> str:
    """
    특정 카테고리의 사용 가능한 전처리 방법들을 문자열로 반환합니다.
    """
    techniques = load_preprocessing_techniques()
    
    if category in techniques:
        tech_list = techniques[category]['techniques']
        return f"=== {category.replace('_', ' ').title()} 전처리 옵션 ===\n" + \
               "\n".join([f"{i+1}. {tech}" for i, tech in enumerate(tech_list)])
    else:
        return f"⚠️ {category} 카테고리의 전처리 방법을 찾을 수 없습니다."


def numeric_planner_node(state: WorkflowState) -> WorkflowState:
    """
    수치형 전처리 플래너 노드
    
    🔍 INPUT STATES:
    - query: 사용자 요청
    - numeric_analysis: 수치형 변수 EDA 결과
    
    📊 OUTPUT STATES:
    - numeric_plan: 수치형 전처리 계획
    
    ➡️ NEXT EDGE: numeric_coder
    """
    print("📋 [PLAN] 수치형 전처리 계획 수립 중...")
    
    try:
        query = state.get("query", "")
        numeric_analysis = state.get("numeric_analysis", {})
        dataframe = state.get("dataframe")
        
        # 실제 데이터프레임의 수치형 컬럼 가져오기
        actual_numeric_columns = dataframe.select_dtypes(include=['number']).columns.tolist()
        
        # KB에서 사용 가능한 전처리 방법들 가져오기
        available_techniques = get_available_techniques("scaling") + "\n\n" + \
                             get_available_techniques("outliers") + "\n\n" + \
                             get_available_techniques("missing_values")

        print("수치형 컬럼명들: ", actual_numeric_columns)
        
        # LLM을 사용하여 전문화된 계획 수립
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=1500)
        
        prompt = f"""
당신은 수치형 데이터 전처리 전문가입니다. 
EDA 분석 결과를 해석하고 인사이트를 도출하여 수치형 변수에 특화된 전처리 계획을 수립해주세요.

=== 실제 데이터프레임 정보 ===
- 전체 컬럼: {list(dataframe.columns)}
- 수치형 컬럼: {actual_numeric_columns}
- 데이터프레임 크기: {dataframe.shape}

=== 사용자 요청 ===
{query}

=== 수치형 EDA 분석 결과 ===
{json.dumps(numeric_analysis, indent=2, ensure_ascii=False)}

=== 분석 결과 해석 및 인사이트 도출 ===
위 EDA 분석 결과를 바탕으로 다음을 수행해주세요:

1. **데이터 분포 특성 분석**:
   - 각 변수의 분포 형태 (정규분포, 치우친 분포, 첨도 등)
   - 이상치 존재 여부와 정도
   - 결측값 패턴

2. **주요 발견사항**:
   - 가장 중요한 데이터 특성들
   - 잠재적 문제점들
   - 모델링에 영향을 줄 수 있는 요소들

3. **전처리 필요성 평가**:
   - 어떤 변수들이 전처리가 필요한지
   - 각 변수별 전처리 우선순위
   - 전처리 방법의 근거

{available_techniques}

=== 응답 형식 ===
{{
    "insights": {{
        "distribution_analysis": "분포 특성 분석 결과",
        "key_findings": "주요 발견사항",
        "data_quality_issues": "데이터 품질 이슈들",
        "preprocessing_needs": "전처리 필요성 평가"
    }},
    "techniques": ["technique1", "technique2"],
    "rationale": "계획 수립 근거 (인사이트 기반)",
    "priority": "high/medium/low",
    "target_columns": {actual_numeric_columns},
    "technique_details": {{
        "technique1": {{
            "columns": {actual_numeric_columns},
            "reason": "이 기법을 선택한 이유",
            "parameters": {{"param1": "value1"}}
        }}
    }}
}}

위에서 제공된 전처리 옵션 중에서만 선택해주세요. JSON 형식으로만 응답해주세요.
실제 데이터프레임의 수치형 컬럼명을 사용하세요: {actual_numeric_columns}
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        
        # JSON 파싱 전에 응답 검증
        try:
            # 응답에서 JSON 부분만 추출
            content = response.content.strip()
            if content.startswith('```json'):
                content = content.split('```json')[1].split('```')[0].strip()
            elif content.startswith('```'):
                content = content.split('```')[1].split('```')[0].strip()
            
            result = json.loads(content)
        except (json.JSONDecodeError, IndexError) as e:
            print(f"⚠️ [PLAN] JSON 파싱 오류, 기본 계획 사용: {e}")
            # 기본 계획 생성
            result = {
                "insights": {
                    "distribution_analysis": "기본 분석 (JSON 파싱 실패)",
                    "key_findings": "기본 발견사항",
                    "data_quality_issues": "기본 품질 이슈",
                    "preprocessing_needs": "기본 전처리 필요성"
                },
                "techniques": ["standard_scaling"],
                "rationale": "기본 수치형 전처리 계획 (JSON 파싱 실패)",
                "priority": "medium",
                "target_columns": [],
                "technique_details": {}
            }
        
        numeric_plan = {
            "domain": "numeric",
            "insights": result.get("insights", {}),
            "techniques": result.get("techniques", []),
            "rationale": result.get("rationale", ""),
            "priority": result.get("priority", "medium"),
            "target_columns": result.get("target_columns", []),
            "technique_details": result.get("technique_details", {}),
            "eda_input": numeric_analysis,
            "status": "success"
        }
        
        print(f"✅ [PLAN] 수치형 계획 완료 - {len(numeric_plan['techniques'])}개 기법 선택")
        
        # 계획 결과 로깅
        print("\n" + "="*50)
        print("📋 [PLAN] 수치형 전처리 계획 결과")
        print("="*50)
        print(f"🎯 우선순위: {numeric_plan['priority']}")
        print(f"🔧 선택된 기법: {', '.join(numeric_plan['techniques'])}")
        if numeric_plan['target_columns']:
            print(f"📊 대상 변수: {', '.join(numeric_plan['target_columns'])}")
        print(f"💡 계획 근거: {numeric_plan['rationale']}")
        
        # 인사이트 로깅
        insights = numeric_plan['insights']
        if insights:
            print(f"\n🔍 주요 인사이트:")
            for key, value in insights.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"   - {key}: {value[:100]}...")
                else:
                    print(f"   - {key}: {value}")
        
        return {
            **state,
            "numeric_plan": numeric_plan
        }
        
    except Exception as e:
        print(f"❌ [PLAN] 수치형 계획 오류: {e}")
        return {
            **state,
            "numeric_plan": {
                "domain": "numeric",
                "insights": {
                    "distribution_analysis": "분석 실패",
                    "key_findings": "분석 실패",
                    "data_quality_issues": "분석 실패",
                    "preprocessing_needs": "분석 실패"
                },
                "techniques": ["standard_scaling"],
                "rationale": f"기본 계획 (오류: {str(e)})",
                "priority": "medium",
                "target_columns": [],
                "technique_details": {},
                "status": "error"
            }
        }


def category_planner_node(state: WorkflowState) -> WorkflowState:
    """
    범주형 전처리 플래너 노드
    
    🔍 INPUT STATES:
    - query: 사용자 요청
    - categorical_analysis: 범주형 변수 EDA 결과
    
    📊 OUTPUT STATES:
    - category_plan: 범주형 전처리 계획
    
    ➡️ NEXT EDGE: category_coder
    """
    print("📋 [PLAN] 범주형 전처리 계획 수립 중...")
    
    try:
        query = state.get("query", "")
        categorical_analysis = state.get("categorical_analysis", {})
        dataframe = state.get("dataframe")
        
        # 실제 데이터프레임의 범주형 컬럼 가져오기
        actual_categorical_columns = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # KB에서 사용 가능한 전처리 방법들 가져오기
        available_techniques = get_available_techniques("categorical_encoding") + "\n\n" + \
                             get_available_techniques("missing_values")
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=1500)
        
        prompt = f"""
당신은 범주형 데이터 전처리 전문가입니다.
EDA 분석 결과를 해석하고 인사이트를 도출하여 범주형 변수에 특화된 전처리 계획을 수립해주세요.

=== 실제 데이터프레임 정보 ===
- 전체 컬럼: {list(dataframe.columns)}
- 범주형 컬럼: {actual_categorical_columns}
- 데이터프레임 크기: {dataframe.shape}

=== 사용자 요청 ===
{query}

=== 범주형 EDA 분석 결과 ===
{json.dumps(categorical_analysis, indent=2, ensure_ascii=False)}

=== 분석 결과 해석 및 인사이트 도출 ===
위 EDA 분석 결과를 바탕으로 다음을 수행해주세요:

1. **범주형 데이터 특성 분석**:
   - 각 변수의 카디널리티 (고유값 개수)
   - 분포 불균형 정도
   - 결측값 패턴
   - 희귀 범주 존재 여부

2. **주요 발견사항**:
   - 가장 중요한 범주형 특성들
   - 잠재적 문제점들 (높은 카디널리티, 불균형 등)
   - 모델링에 영향을 줄 수 있는 요소들

3. **전처리 필요성 평가**:
   - 어떤 변수들이 전처리가 필요한지
   - 각 변수별 전처리 우선순위
   - 전처리 방법의 근거

{available_techniques}

=== 응답 형식 ===
{{
    "insights": {{
        "cardinality_analysis": "카디널리티 분석 결과",
        "distribution_analysis": "분포 특성 분석 결과",
        "key_findings": "주요 발견사항",
        "data_quality_issues": "데이터 품질 이슈들",
        "preprocessing_needs": "전처리 필요성 평가"
    }},
    "techniques": ["technique1", "technique2"],
    "rationale": "계획 수립 근거 (인사이트 기반)",
    "priority": "high/medium/low",
    "target_columns": {actual_categorical_columns},
    "technique_details": {{
        "technique1": {{
            "columns": {actual_categorical_columns},
            "reason": "이 기법을 선택한 이유",
            "parameters": {{"param1": "value1"}}
        }}
    }}
}}

위에서 제공된 전처리 옵션 중에서만 선택해주세요. JSON 형식으로만 응답해주세요.
실제 데이터프레임의 범주형 컬럼명을 사용하세요: {actual_categorical_columns}
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        
        # JSON 파싱 전에 응답 검증
        try:
            # 응답에서 JSON 부분만 추출
            content = response.content.strip()
            if content.startswith('```json'):
                content = content.split('```json')[1].split('```')[0].strip()
            elif content.startswith('```'):
                content = content.split('```')[1].split('```')[0].strip()
            
            result = json.loads(content)
        except (json.JSONDecodeError, IndexError) as e:
            print(f"⚠️ [PLAN] JSON 파싱 오류, 기본 계획 사용: {e}")
            # 기본 계획 생성
            result = {
                "insights": {
                    "cardinality_analysis": "기본 분석 (JSON 파싱 실패)",
                    "distribution_analysis": "기본 분석",
                    "key_findings": "기본 발견사항",
                    "data_quality_issues": "기본 품질 이슈",
                    "preprocessing_needs": "기본 전처리 필요성"
                },
                "techniques": ["label_encoding"],
                "rationale": "기본 범주형 전처리 계획 (JSON 파싱 실패)",
                "priority": "medium",
                "target_columns": [],
                "technique_details": {}
            }
        
        category_plan = {
            "domain": "category",
            "insights": result.get("insights", {}),
            "techniques": result.get("techniques", []),
            "rationale": result.get("rationale", ""),
            "priority": result.get("priority", "medium"),
            "target_columns": result.get("target_columns", []),
            "technique_details": result.get("technique_details", {}),
            "eda_input": categorical_analysis,
            "status": "success"
        }
        
        print(f"✅ [PLAN] 범주형 계획 완료 - {len(category_plan['techniques'])}개 기법 선택")
        
        # 계획 결과 로깅
        print("\n" + "="*50)
        print("📋 [PLAN] 범주형 전처리 계획 결과")
        print("="*50)
        print(f"🎯 우선순위: {category_plan['priority']}")
        print(f"🔧 선택된 기법: {', '.join(category_plan['techniques'])}")
        if category_plan['target_columns']:
            print(f"📊 대상 변수: {', '.join(category_plan['target_columns'])}")
        print(f"💡 계획 근거: {category_plan['rationale']}")
        
        # 인사이트 로깅
        insights = category_plan['insights']
        if insights:
            print(f"\n🔍 주요 인사이트:")
            for key, value in insights.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"   - {key}: {value[:100]}...")
                else:
                    print(f"   - {key}: {value}")
        
        return {
            **state,
            "category_plan": category_plan
        }
        
    except Exception as e:
        print(f"❌ [PLAN] 범주형 계획 오류: {e}")
        return {
            **state,
            "category_plan": {
                "domain": "category",
                "insights": {
                    "cardinality_analysis": "분석 실패",
                    "distribution_analysis": "분석 실패",
                    "key_findings": "분석 실패",
                    "data_quality_issues": "분석 실패",
                    "preprocessing_needs": "분석 실패"
                },
                "techniques": ["label_encoding"],
                "rationale": f"기본 계획 (오류: {str(e)})",
                "priority": "medium",
                "target_columns": [],
                "technique_details": {},
                "status": "error"
            }
        }


def outlier_planner_node(state: WorkflowState) -> WorkflowState:
    """
    이상치 전처리 플래너 노드
    
    🔍 INPUT STATES:
    - query: 사용자 요청
    - outlier_analysis: 이상치 EDA 결과
    
    📊 OUTPUT STATES:
    - outlier_plan: 이상치 전처리 계획
    
    ➡️ NEXT EDGE: outlier_coder
    """
    print("📋 [PLAN] 이상치 전처리 계획 수립 중...")
    
    try:
        query = state.get("query", "")
        outlier_analysis = state.get("outlier_analysis", {})
        dataframe = state.get("dataframe")
        
        # 실제 데이터프레임의 수치형 컬럼 가져오기 (이상치는 수치형에만 적용)
        actual_numeric_columns = dataframe.select_dtypes(include=['number']).columns.tolist()
        
        # KB에서 사용 가능한 전처리 방법들 가져오기
        available_techniques = get_available_techniques("outliers")
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=1500)
        
        prompt = f"""
당신은 이상치 처리 전문가입니다.
EDA 분석 결과를 해석하고 인사이트를 도출하여 이상치 처리 계획을 수립해주세요.

=== 실제 데이터프레임 정보 ===
- 전체 컬럼: {list(dataframe.columns)}
- 수치형 컬럼 (이상치 처리 대상): {actual_numeric_columns}
- 데이터프레임 크기: {dataframe.shape}

=== 사용자 요청 ===
{query}

=== 이상치 EDA 분석 결과 ===
{json.dumps(outlier_analysis, indent=2, ensure_ascii=False)}

=== 분석 결과 해석 및 인사이트 도출 ===
위 EDA 분석 결과를 바탕으로 다음을 수행해주세요:

1. **이상치 패턴 분석**:
   - IQR과 Z-Score 방법의 차이점
   - 각 변수별 이상치 분포 특성
   - 이상치의 심각도 (비율, 범위 등)

2. **주요 발견사항**:
   - 가장 문제가 되는 이상치 패턴들
   - 이상치가 발생할 수 있는 원인 추정
   - 모델링에 미칠 수 있는 영향

3. **처리 전략 평가**:
   - 어떤 변수들이 이상치 처리가 필요한지
   - 각 변수별 처리 방법의 적합성
   - 처리 우선순위

{available_techniques}

=== 응답 형식 ===
{{
    "insights": {{
        "outlier_pattern_analysis": "이상치 패턴 분석 결과",
        "severity_assessment": "이상치 심각도 평가",
        "key_findings": "주요 발견사항",
        "potential_causes": "이상치 발생 가능 원인",
        "modeling_impact": "모델링에 미칠 영향"
    }},
    "techniques": ["technique1", "technique2"],
    "rationale": "계획 수립 근거 (인사이트 기반)",
    "priority": "high/medium/low",
    "target_columns": {actual_numeric_columns},
    "technique_details": {{
        "technique1": {{
            "columns": {actual_numeric_columns},
            "reason": "이 기법을 선택한 이유",
            "parameters": {{"param1": "value1"}}
        }}
    }}
}}

위에서 제공된 전처리 옵션 중에서만 선택해주세요. JSON 형식으로만 응답해주세요.
실제 데이터프레임의 수치형 컬럼명을 사용하세요: {actual_numeric_columns}
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        
        # JSON 파싱 전에 응답 검증
        try:
            # 응답에서 JSON 부분만 추출
            content = response.content.strip()
            if content.startswith('```json'):
                content = content.split('```json')[1].split('```')[0].strip()
            elif content.startswith('```'):
                content = content.split('```')[1].split('```')[0].strip()
            
            result = json.loads(content)
        except (json.JSONDecodeError, IndexError) as e:
            print(f"⚠️ [PLAN] JSON 파싱 오류, 기본 계획 사용: {e}")
            # 기본 계획 생성
            result = {
                "insights": {
                    "outlier_pattern_analysis": "기본 분석 (JSON 파싱 실패)",
                    "severity_assessment": "기본 평가",
                    "key_findings": "기본 발견사항",
                    "potential_causes": "기본 원인",
                    "modeling_impact": "기본 영향"
                },
                "techniques": ["iqr_outlier_detection"],
                "rationale": "기본 이상치 처리 계획 (JSON 파싱 실패)",
                "priority": "medium",
                "target_columns": [],
                "technique_details": {}
            }
        
        outlier_plan = {
            "domain": "outlier",
            "insights": result.get("insights", {}),
            "techniques": result.get("techniques", []),
            "rationale": result.get("rationale", ""),
            "priority": result.get("priority", "medium"),
            "target_columns": result.get("target_columns", []),
            "technique_details": result.get("technique_details", {}),
            "eda_input": outlier_analysis,
            "status": "success"
        }
        
        print(f"✅ [PLAN] 이상치 계획 완료 - {len(outlier_plan['techniques'])}개 기법 선택")
        
        # 계획 결과 로깅
        print("\n" + "="*50)
        print("📋 [PLAN] 이상치 전처리 계획 결과")
        print("="*50)
        print(f"🎯 우선순위: {outlier_plan['priority']}")
        print(f"🔧 선택된 기법: {', '.join(outlier_plan['techniques'])}")
        if outlier_plan['target_columns']:
            print(f"📊 대상 변수: {', '.join(outlier_plan['target_columns'])}")
        print(f"💡 계획 근거: {outlier_plan['rationale']}")
        
        # 인사이트 로깅
        insights = outlier_plan['insights']
        if insights:
            print(f"\n🔍 주요 인사이트:")
            for key, value in insights.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"   - {key}: {value[:100]}...")
                else:
                    print(f"   - {key}: {value}")
        
        return {
            **state,
            "outlier_plan": outlier_plan
        }
        
    except Exception as e:
        print(f"❌ [PLAN] 이상치 계획 오류: {e}")
        return {
            **state,
            "outlier_plan": {
                "domain": "outlier",
                "insights": {
                    "outlier_pattern_analysis": "분석 실패",
                    "severity_assessment": "평가 실패",
                    "key_findings": "분석 실패",
                    "potential_causes": "분석 실패",
                    "modeling_impact": "분석 실패"
                },
                "techniques": ["iqr_outlier_detection"],
                "rationale": f"기본 계획 (오류: {str(e)})",
                "priority": "medium",
                "target_columns": [],
                "technique_details": {},
                "status": "error"
            }
        }


def nulldata_planner_node(state: WorkflowState) -> WorkflowState:
    """
    결측치 전처리 플래너 노드
    
    🔍 INPUT STATES:
    - query: 사용자 요청
    - missing_duplicate_analysis: 결측치 및 중복 데이터 EDA 결과
    
    📊 OUTPUT STATES:
    - nulldata_plan: 결측치 전처리 계획
    
    ➡️ NEXT EDGE: nulldata_coder
    """
    print("📋 [PLAN] 결측치 전처리 계획 수립 중...")
    
    try:
        query = state.get("query", "")
        missing_duplicate_analysis = state.get("missing_duplicate_analysis", {})
        dataframe = state.get("dataframe")
        
        # 실제 데이터프레임의 결측값이 있는 컬럼 가져오기
        actual_missing_columns = dataframe.columns[dataframe.isnull().any()].tolist()
        
        # KB에서 사용 가능한 전처리 방법들 가져오기
        available_techniques = get_available_techniques("missing_values")
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=1500)
        
        prompt = f"""
당신은 결측치 및 중복 데이터 처리 전문가입니다.
EDA 분석 결과를 해석하고 인사이트를 도출하여 결측치 처리 계획을 수립해주세요.

=== 실제 데이터프레임 정보 ===
- 전체 컬럼: {list(dataframe.columns)}
- 결측값이 있는 컬럼: {actual_missing_columns}
- 데이터프레임 크기: {dataframe.shape}
- 총 결측값 수: {dataframe.isnull().sum().sum()}개

=== 사용자 요청 ===
{query}

=== 결측치 및 중복 데이터 EDA 분석 결과 ===
{json.dumps(missing_duplicate_analysis, indent=2, ensure_ascii=False)}

=== 분석 결과 해석 및 인사이트 도출 ===
위 EDA 분석 결과를 바탕으로 다음을 수행해주세요:

1. **결측치 패턴 분석**:
   - 전체 결측치 비율과 심각도
   - 컬럼별 결측치 분포 특성
   - 결측치 패턴 (무작위 vs 체계적)
   - 중복 데이터의 특성과 영향

2. **주요 발견사항**:
   - 가장 문제가 되는 결측치 패턴들
   - 결측치가 발생할 수 있는 원인 추정
   - 데이터 품질 이슈들

3. **처리 전략 평가**:
   - 어떤 변수들이 결측치 처리가 필요한지
   - 각 변수별 처리 방법의 적합성
   - 처리 우선순위

{available_techniques}

=== 응답 형식 ===
{{
    "insights": {{
        "missing_pattern_analysis": "결측치 패턴 분석 결과",
        "data_quality_assessment": "데이터 품질 평가",
        "key_findings": "주요 발견사항",
        "potential_causes": "결측치 발생 가능 원인",
        "duplicate_analysis": "중복 데이터 분석"
    }},
    "techniques": ["technique1", "technique2"],
    "rationale": "계획 수립 근거 (인사이트 기반)",
    "priority": "high/medium/low",
    "target_columns": {actual_missing_columns},
    "technique_details": {{
        "technique1": {{
            "columns": {actual_missing_columns},
            "reason": "이 기법을 선택한 이유",
            "parameters": {{"param1": "value1"}}
        }}
    }}
}}

위에서 제공된 전처리 옵션 중에서만 선택해주세요. JSON 형식으로만 응답해주세요.
실제 데이터프레임의 컬럼명을 사용하세요: {list(dataframe.columns)}
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        
        # JSON 파싱 전에 응답 검증
        try:
            # 응답에서 JSON 부분만 추출
            content = response.content.strip()
            if content.startswith('```json'):
                content = content.split('```json')[1].split('```')[0].strip()
            elif content.startswith('```'):
                content = content.split('```')[1].split('```')[0].strip()
            
            result = json.loads(content)
        except (json.JSONDecodeError, IndexError) as e:
            print(f"⚠️ [PLAN] JSON 파싱 오류, 기본 계획 사용: {e}")
            # 기본 계획 생성
            result = {
                "insights": {
                    "missing_pattern_analysis": "기본 분석 (JSON 파싱 실패)",
                    "data_quality_assessment": "기본 평가",
                    "key_findings": "기본 발견사항",
                    "potential_causes": "기본 원인",
                    "duplicate_analysis": "기본 분석"
                },
                "techniques": ["fill_numerical_median"],
                "rationale": "기본 결측치 처리 계획 (JSON 파싱 실패)",
                "priority": "high",
                "target_columns": [],
                "technique_details": {}
            }
        
        nulldata_plan = {
            "domain": "nulldata",
            "insights": result.get("insights", {}),
            "techniques": result.get("techniques", []),
            "rationale": result.get("rationale", ""),
            "priority": result.get("priority", "high"),
            "target_columns": result.get("target_columns", []),
            "technique_details": result.get("technique_details", {}),
            "eda_input": missing_duplicate_analysis,
            "status": "success"
        }
        
        print(f"✅ [PLAN] 결측치 계획 완료 - {len(nulldata_plan['techniques'])}개 기법 선택")
        
        # 계획 결과 로깅
        print("\n" + "="*50)
        print("📋 [PLAN] 결측치 전처리 계획 결과")
        print("="*50)
        print(f"🎯 우선순위: {nulldata_plan['priority']}")
        print(f"🔧 선택된 기법: {', '.join(nulldata_plan['techniques'])}")
        if nulldata_plan['target_columns']:
            print(f"📊 대상 변수: {', '.join(nulldata_plan['target_columns'])}")
        print(f"💡 계획 근거: {nulldata_plan['rationale']}")
        
        # 인사이트 로깅
        insights = nulldata_plan['insights']
        if insights:
            print(f"\n🔍 주요 인사이트:")
            for key, value in insights.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"   - {key}: {value[:100]}...")
                else:
                    print(f"   - {key}: {value}")
        
        return {
            **state,
            "nulldata_plan": nulldata_plan
        }
        
    except Exception as e:
        print(f"❌ [PLAN] 결측치 계획 오류: {e}")
        return {
            **state,
            "nulldata_plan": {
                "domain": "nulldata",
                "insights": {
                    "missing_pattern_analysis": "분석 실패",
                    "data_quality_assessment": "평가 실패",
                    "key_findings": "분석 실패",
                    "potential_causes": "분석 실패",
                    "duplicate_analysis": "분석 실패"
                },
                "techniques": ["fill_numerical_median"],
                "rationale": f"기본 계획 (오류: {str(e)})",
                "priority": "high",
                "target_columns": [],
                "technique_details": {},
                "status": "error"
            }
        }


def corr_planner_node(state: WorkflowState) -> WorkflowState:
    """
    상관관계 전처리 플래너 노드
    
    🔍 INPUT STATES:
    - query: 사용자 요청
    - correlation_analysis: 상관관계 EDA 결과
    
    📊 OUTPUT STATES:
    - corr_plan: 상관관계 전처리 계획
    
    ➡️ NEXT EDGE: corr_coder
    """
    print("📋 [PLAN] 상관관계 전처리 계획 수립 중...")
    
    try:
        query = state.get("query", "")
        correlation_analysis = state.get("correlation_analysis", {})
        dataframe = state.get("dataframe")
        
        # 실제 데이터프레임의 수치형 컬럼 가져오기 (상관관계는 수치형에만 적용)
        actual_numeric_columns = dataframe.select_dtypes(include=['number']).columns.tolist()
        
        # KB에서 사용 가능한 전처리 방법들 가져오기
        available_techniques = get_available_techniques("feature_selection") + "\n\n" + \
                             get_available_techniques("dimensionality_reduction")
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=1500)
        
        prompt = f"""
당신은 상관관계 및 특성 선택 전문가입니다.
EDA 분석 결과를 해석하고 인사이트를 도출하여 상관관계 기반 전처리 계획을 수립해주세요.

=== 실제 데이터프레임 정보 ===
- 전체 컬럼: {list(dataframe.columns)}
- 수치형 컬럼 (상관관계 분석 대상): {actual_numeric_columns}
- 데이터프레임 크기: {dataframe.shape}

=== 사용자 요청 ===
{query}

=== 상관관계 EDA 분석 결과 ===
{json.dumps(correlation_analysis, indent=2, ensure_ascii=False)}

=== 분석 결과 해석 및 인사이트 도출 ===
위 EDA 분석 결과를 바탕으로 다음을 수행해주세요:

1. **상관관계 패턴 분석**:
   - 전체 상관관계 분포 특성
   - 강한 상관관계 변수 쌍들
   - 다중공선성 의심 변수들
   - 타겟 변수와의 상관관계

2. **주요 발견사항**:
   - 가장 중요한 상관관계 패턴들
   - 잠재적 문제점들 (다중공선성, 약한 상관관계 등)
   - 모델링에 영향을 줄 수 있는 요소들

3. **특성 선택 전략 평가**:
   - 어떤 변수들이 제거/선택되어야 하는지
   - 각 변수별 선택/제거 근거
   - 특성 선택 우선순위

{available_techniques}

=== 응답 형식 ===
{{
    "insights": {{
        "correlation_pattern_analysis": "상관관계 패턴 분석 결과",
        "multicollinearity_assessment": "다중공선성 평가",
        "key_findings": "주요 발견사항",
        "feature_importance": "특성 중요도 분석",
        "modeling_impact": "모델링에 미칠 영향"
    }},
    "techniques": ["technique1", "technique2"],
    "rationale": "계획 수립 근거 (인사이트 기반)",
    "priority": "high/medium/low",
    "target_columns": {actual_numeric_columns},
    "technique_details": {{
        "technique1": {{
            "columns": {actual_numeric_columns},
            "reason": "이 기법을 선택한 이유",
            "parameters": {{"param1": "value1"}}
        }}
    }}
}}

위에서 제공된 전처리 옵션 중에서만 선택해주세요. JSON 형식으로만 응답해주세요.
실제 데이터프레임의 수치형 컬럼명을 사용하세요: {actual_numeric_columns}
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        
        # JSON 파싱 전에 응답 검증
        try:
            # 응답에서 JSON 부분만 추출
            content = response.content.strip()
            if content.startswith('```json'):
                content = content.split('```json')[1].split('```')[0].strip()
            elif content.startswith('```'):
                content = content.split('```')[1].split('```')[0].strip()
            
            result = json.loads(content)
        except (json.JSONDecodeError, IndexError) as e:
            print(f"⚠️ [PLAN] JSON 파싱 오류, 기본 계획 사용: {e}")
            # 기본 계획 생성
            result = {
                "insights": {
                    "correlation_pattern_analysis": "기본 분석 (JSON 파싱 실패)",
                    "multicollinearity_assessment": "기본 평가",
                    "key_findings": "기본 발견사항",
                    "feature_importance": "기본 분석",
                    "modeling_impact": "기본 영향"
                },
                "techniques": ["correlation_filter"],
                "rationale": "기본 상관관계 처리 계획 (JSON 파싱 실패)",
                "priority": "low",
                "target_columns": [],
                "technique_details": {}
            }
        
        corr_plan = {
            "domain": "correlation",
            "insights": result.get("insights", {}),
            "techniques": result.get("techniques", []),
            "rationale": result.get("rationale", ""),
            "priority": result.get("priority", "low"),  # 상관관계 처리는 보통 낮은 우선순위
            "target_columns": result.get("target_columns", []),
            "technique_details": result.get("technique_details", {}),
            "eda_input": correlation_analysis,
            "status": "success"
        }
        
        print(f"✅ [PLAN] 상관관계 계획 완료 - {len(corr_plan['techniques'])}개 기법 선택")
        
        # 계획 결과 로깅
        print("\n" + "="*50)
        print("📋 [PLAN] 상관관계 전처리 계획 결과")
        print("="*50)
        print(f"🎯 우선순위: {corr_plan['priority']}")
        print(f"🔧 선택된 기법: {', '.join(corr_plan['techniques'])}")
        if corr_plan['target_columns']:
            print(f"📊 대상 변수: {', '.join(corr_plan['target_columns'])}")
        print(f"💡 계획 근거: {corr_plan['rationale']}")
        
        # 인사이트 로깅
        insights = corr_plan['insights']
        if insights:
            print(f"\n🔍 주요 인사이트:")
            for key, value in insights.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"   - {key}: {value[:100]}...")
                else:
                    print(f"   - {key}: {value}")
        
        return {
            **state,
            "corr_plan": corr_plan
        }
        
    except Exception as e:
        print(f"❌ [PLAN] 상관관계 계획 오류: {e}")
        return {
            **state,
            "corr_plan": {
                "domain": "correlation",
                "insights": {
                    "correlation_pattern_analysis": "분석 실패",
                    "multicollinearity_assessment": "평가 실패",
                    "key_findings": "분석 실패",
                    "feature_importance": "분석 실패",
                    "modeling_impact": "분석 실패"
                },
                "techniques": ["correlation_filter"],
                "rationale": f"기본 계획 (오류: {str(e)})",
                "priority": "low",
                "target_columns": [],
                "technique_details": {},
                "status": "error"
            }
        } 