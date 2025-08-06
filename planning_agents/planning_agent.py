"""
Planning Agent
EDA 결과를 분석하여 전처리 작업 계획을 수립하는 에이전트입니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda


class PlanningAgent:
    """
    Planning Agent
    EDA 결과를 분석하여 전처리 작업의 우선순위와 순서를 결정합니다.
    """
    
    def __init__(self):
        """Planning Agent 초기화"""
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=1500
        )
        
        # 전처리 작업 카테고리 정의
        self.preprocessing_categories = {
            'missing_values': {
                'description': '결측값 처리',
                'priority': 1,
                'dependencies': [],
                'keywords': ['missing', 'null', 'nan', '결측']
            },
            'outliers': {
                'description': '이상치 처리',
                'priority': 2,
                'dependencies': ['missing_values'],
                'keywords': ['outlier', '이상치', 'extreme', 'iqr', 'zscore']
            },
            'duplicates': {
                'description': '중복 데이터 처리',
                'priority': 1,
                'dependencies': [],
                'keywords': ['duplicate', '중복', 'repeated']
            },
            'categorical_encoding': {
                'description': '범주형 변수 인코딩',
                'priority': 3,
                'dependencies': ['missing_values'],
                'keywords': ['categorical', '범주형', 'object', 'string', 'encoding']
            },
            'scaling': {
                'description': '특성 스케일링',
                'priority': 4,
                'dependencies': ['missing_values', 'outliers'],
                'keywords': ['scaling', 'normalization', '스케일링', '정규화']
            },
            'feature_selection': {
                'description': '특성 선택',
                'priority': 5,
                'dependencies': ['missing_values', 'categorical_encoding'],
                'keywords': ['feature selection', '특성 선택', 'variance', 'correlation']
            },
            'feature_engineering': {
                'description': '특성 엔지니어링',
                'priority': 6,
                'dependencies': ['missing_values', 'categorical_encoding'],
                'keywords': ['feature engineering', '특성 엔지니어링', 'polynomial', 'interaction']
            },
            'dimensionality_reduction': {
                'description': '차원 축소',
                'priority': 7,
                'dependencies': ['feature_selection', 'feature_engineering'],
                'keywords': ['dimensionality reduction', '차원 축소', 'pca', 'tsne']
            },
            'class_imbalance': {
                'description': '클래스 불균형 처리',
                'priority': 8,
                'dependencies': ['missing_values', 'categorical_encoding'],
                'keywords': ['imbalance', '불균형', 'smote', 'undersampling']
            }
        }
        
        print("Planning Agent 초기화 완료")
    
    def analyze_eda_results(self, eda_results: Dict[str, Any]) -> Dict[str, Any]:
        """EDA 결과를 분석하여 전처리 필요성을 판단"""
        
        analysis = {
            'missing_values_needed': False,
            'outliers_needed': False,
            'duplicates_needed': False,
            'categorical_encoding_needed': False,
            'scaling_needed': False,
            'feature_selection_needed': False,
            'feature_engineering_needed': False,
            'dimensionality_reduction_needed': False,
            'class_imbalance_needed': False,
            'data_characteristics': {},
            'recommendations': []
        }
        
        # 결측값 분석
        null_analysis = eda_results.get('null_analysis_text', '')
        if 'missing' in null_analysis.lower() or 'null' in null_analysis.lower():
            analysis['missing_values_needed'] = True
            analysis['recommendations'].append('결측값 처리가 필요합니다.')
        
        # 이상치 분석
        outlier_analysis = eda_results.get('outlier_analysis_text', '')
        if 'outlier' in outlier_analysis.lower() or '이상치' in outlier_analysis:
            analysis['outliers_needed'] = True
            analysis['recommendations'].append('이상치 처리가 필요합니다.')
        
        # 범주형 변수 분석
        cate_analysis = eda_results.get('cate_analysis_text', '')
        if 'categorical' in cate_analysis.lower() or '범주형' in cate_analysis:
            analysis['categorical_encoding_needed'] = True
            analysis['recommendations'].append('범주형 변수 인코딩이 필요합니다.')
        
        # 수치형 변수 분석
        numeric_analysis = eda_results.get('numeric_analysis_text', '')
        if 'scaling' in numeric_analysis.lower() or '스케일링' in numeric_analysis:
            analysis['scaling_needed'] = True
            analysis['recommendations'].append('특성 스케일링이 필요합니다.')
        
        # 상관관계 분석
        corr_analysis = eda_results.get('corr_analysis_text', '')
        if 'correlation' in corr_analysis.lower() or '상관관계' in corr_analysis:
            analysis['feature_selection_needed'] = True
            analysis['recommendations'].append('특성 선택이 필요합니다.')
        
        # 전체 데이터 분석
        text_analysis = eda_results.get('text_analysis', '')
        
        # 데이터 크기 및 특성 분석
        if 'large' in text_analysis.lower() or 'many features' in text_analysis.lower():
            analysis['dimensionality_reduction_needed'] = True
            analysis['recommendations'].append('차원 축소가 필요합니다.')
        
        if 'imbalance' in text_analysis.lower() or '불균형' in text_analysis:
            analysis['class_imbalance_needed'] = True
            analysis['recommendations'].append('클래스 불균형 처리가 필요합니다.')
        
        return analysis
    
    def create_preprocessing_plan(self, eda_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """전처리 계획 수립"""
        
        # 필요한 작업들 식별
        needed_tasks = []
        for task, needed in eda_analysis.items():
            if needed and task.endswith('_needed'):
                task_name = task.replace('_needed', '')
                needed_tasks.append(task_name)
        
        # 우선순위에 따른 작업 순서 결정
        prioritized_tasks = self.prioritize_tasks(needed_tasks)
        
        # 의존성 검사 및 순서 조정
        final_order = self.check_dependencies(prioritized_tasks)
        
        # 계획 생성
        plan = {
            'tasks': final_order,
            'priority': final_order,
            'rationale': self.generate_rationale(eda_analysis, final_order),
            'estimated_steps': len(final_order),
            'complexity': self.assess_complexity(final_order),
            'time_estimate': self.estimate_time(final_order)
        }
        
        return plan
    
    def prioritize_tasks(self, tasks: List[str]) -> List[str]:
        """작업 우선순위 결정"""
        if not tasks:
            return ['missing_values', 'categorical_encoding', 'scaling']
        
        # 우선순위 점수 계산
        task_scores = []
        for task in tasks:
            if task in self.preprocessing_categories:
                score = self.preprocessing_categories[task]['priority']
                task_scores.append((task, score))
        
        # 우선순위 순으로 정렬
        task_scores.sort(key=lambda x: x[1])
        return [task for task, score in task_scores]
    
    def check_dependencies(self, tasks: List[str]) -> List[str]:
        """의존성 검사 및 순서 조정"""
        final_order = []
        added_tasks = set()
        
        for task in tasks:
            if task in self.preprocessing_categories:
                dependencies = self.preprocessing_categories[task]['dependencies']
                
                # 의존성 작업들을 먼저 추가
                for dep in dependencies:
                    if dep not in added_tasks and dep in tasks:
                        if dep not in final_order:
                            final_order.append(dep)
                            added_tasks.add(dep)
                
                # 현재 작업 추가
                if task not in added_tasks:
                    final_order.append(task)
                    added_tasks.add(task)
        
        return final_order
    
    def generate_rationale(self, eda_analysis: Dict[str, Any], tasks: List[str]) -> str:
        """계획에 대한 근거 생성"""
        rationale_parts = []
        
        if 'missing_values' in tasks:
            rationale_parts.append("결측값을 먼저 처리하여 데이터 품질을 확보")
        
        if 'outliers' in tasks:
            rationale_parts.append("이상치를 처리하여 모델 성능에 영향을 주는 극값들을 제거")
        
        if 'categorical_encoding' in tasks:
            rationale_parts.append("범주형 변수를 수치형으로 변환하여 머신러닝 모델이 처리할 수 있도록 함")
        
        if 'scaling' in tasks:
            rationale_parts.append("특성 스케일링을 통해 모든 변수가 동일한 스케일을 가지도록 정규화")
        
        if 'feature_selection' in tasks:
            rationale_parts.append("불필요한 특성을 제거하여 모델 복잡도를 줄이고 성능을 향상")
        
        if 'feature_engineering' in tasks:
            rationale_parts.append("새로운 특성을 생성하여 모델의 예측 능력을 향상")
        
        if 'dimensionality_reduction' in tasks:
            rationale_parts.append("차원을 축소하여 계산 효율성을 높이고 차원의 저주 문제 해결")
        
        if 'class_imbalance' in tasks:
            rationale_parts.append("클래스 불균형을 해결하여 모델이 모든 클래스를 균등하게 학습하도록 함")
        
        return ". ".join(rationale_parts) if rationale_parts else "표준 전처리 파이프라인 적용"
    
    def assess_complexity(self, tasks: List[str]) -> str:
        """작업 복잡도 평가"""
        if len(tasks) <= 3:
            return "Low"
        elif len(tasks) <= 5:
            return "Medium"
        else:
            return "High"
    
    def estimate_time(self, tasks: List[str]) -> str:
        """예상 소요 시간 추정"""
        base_time = len(tasks) * 2  # 기본 2분/작업
        if len(tasks) <= 3:
            return f"{base_time}분 (간단한 전처리)"
        elif len(tasks) <= 5:
            return f"{base_time}분 (중간 복잡도 전처리)"
        else:
            return f"{base_time}분 (복잡한 전처리)"
    
    def generate_planning_prompt(self, eda_results: Dict[str, Any]) -> str:
        """Planning을 위한 LLM 프롬프트 생성"""
        
        prompt = f"""
You are a data preprocessing planning expert. Analyze the EDA results and create a comprehensive preprocessing plan.

=== EDA Analysis Results ===
Text Analysis: {eda_results.get('text_analysis', '')}
Missing Value Analysis: {eda_results.get('null_analysis_text', '')}
Outlier Analysis: {eda_results.get('outlier_analysis_text', '')}
Categorical Analysis: {eda_results.get('cate_analysis_text', '')}
Numeric Analysis: {eda_results.get('numeric_analysis_text', '')}
Correlation Analysis: {eda_results.get('corr_analysis_text', '')}

=== Available Preprocessing Tasks ===
1. missing_values - Handle missing data
2. outliers - Detect and handle outliers
3. duplicates - Remove duplicate rows
4. categorical_encoding - Encode categorical variables
5. scaling - Scale numerical features
6. feature_selection - Select important features
7. feature_engineering - Create new features
8. dimensionality_reduction - Reduce dimensions
9. class_imbalance - Handle class imbalance

=== Requirements ===
1. Analyze the EDA results to identify which preprocessing tasks are needed
2. Consider the logical order of preprocessing steps
3. Prioritize tasks based on data quality issues
4. Consider dependencies between tasks
5. Provide rationale for the chosen order

Please provide a JSON response with the following structure:
{{
    "tasks": ["task1", "task2", ...],
    "priority": ["task1", "task2", ...],
    "rationale": "explanation of the plan",
    "complexity": "Low/Medium/High",
    "estimated_time": "time estimate"
}}
"""
        return prompt
    
    def create_plan_with_llm(self, eda_results: Dict[str, Any]) -> Dict[str, Any]:
        """LLM을 사용하여 전처리 계획 생성"""
        
        try:
            prompt = self.generate_planning_prompt(eda_results)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # JSON 응답 파싱 시도
            import json
            try:
                plan = json.loads(response.content)
                return plan
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 기본 계획 사용
                print("LLM 응답을 JSON으로 파싱할 수 없어 기본 계획을 사용합니다.")
                return self.create_basic_plan(eda_results)
        
        except Exception as e:
            print(f"LLM 계획 생성 오류: {e}")
            return self.create_basic_plan(eda_results)
    
    def create_basic_plan(self, eda_results: Dict[str, Any]) -> Dict[str, Any]:
        """기본 전처리 계획 생성"""
        analysis = self.analyze_eda_results(eda_results)
        plan = self.create_preprocessing_plan(analysis)
        return plan


def planning_agent_function(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Planning Agent 함수
    EDA 결과를 분석하여 전처리 계획을 수립합니다.
    """
    try:
        # Planning Agent 인스턴스 생성
        planning_agent = PlanningAgent()
        
        # EDA 결과 추출
        eda_results = {
            'text_analysis': inputs.get('text_analysis', ''),
            'null_analysis_text': inputs.get('null_analysis_text', ''),
            'outlier_analysis_text': inputs.get('outlier_analysis_text', ''),
            'cate_analysis_text': inputs.get('cate_analysis_text', ''),
            'numeric_analysis_text': inputs.get('numeric_analysis_text', ''),
            'corr_analysis_text': inputs.get('corr_analysis_text', ''),
        }
        
        # LLM을 사용한 계획 생성
        planning_info = planning_agent.create_plan_with_llm(eda_results)
        
        print(f"Planning Agent: 전처리 계획 수립 완료")
        print(f"  작업 목록: {planning_info.get('tasks', [])}")
        print(f"  우선순위: {planning_info.get('priority', [])}")
        print(f"  복잡도: {planning_info.get('complexity', 'Unknown')}")
        
        return {
            **inputs,
            'planning_info': planning_info
        }
        
    except Exception as e:
        print(f"Planning Agent 오류: {e}")
        # 기본 계획 반환
        default_plan = {
            'tasks': ['missing_values', 'outliers', 'categorical_encoding', 'scaling'],
            'priority': ['missing_values', 'outliers', 'categorical_encoding', 'scaling'],
            'rationale': '기본 전처리 파이프라인',
            'complexity': 'Medium',
            'estimated_time': '8분'
        }
        
        return {
            **inputs,
            'planning_info': default_plan,
            'error': str(e)
        }


# LangGraph 노드로 사용할 수 있는 함수
planning_agent = RunnableLambda(planning_agent_function)