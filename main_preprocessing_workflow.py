#!/usr/bin/env python3
"""
전처리 워크플로우 메인 오케스트레이터

Flow:
1. DataFrame 입력
2. EDA 진행 (text + image)
3. EDA 정보를 각 preprocessing agents에게 전달
4. MultiModal LLM이 Knowledge Base에서 적절한 전처리 방법 선택
5. LLM Coding Agent가 실제 전처리 코드 작성
6. 여러 전처리 코드를 통합하여 최종 전처리 실행
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Import agents
from agents.eda_agents.numeric_agent.numeric_text import NumericTextAgent
from agents.eda_agents.numeric_agent.numeric_image import NumericImageAgent
from agents.eda_agents.category_agent.cate_text import CategoryTextAgent
from agents.eda_agents.category_agent.cate_image import CategoryImageAgent
from agents.eda_agents.nulldata_agent.null_text import NullDataTextAgent
from agents.eda_agents.nulldata_agent.null_image import NullDataImageAgent
from agents.eda_agents.corr_agent.corr_text import CorrelationTextAgent
from agents.eda_agents.corr_agent.corr_image import CorrelationImageAgent
from agents.eda_agents.outlier_agent.outlier_text import OutlierTextAgent
from agents.eda_agents.outlier_agent.outlier_image import OutlierImageAgent

# Import preprocessing agents
from agents.preprocessing_agents.basic.nulldata_agent import NullDataPreprocessingAgent
from agents.preprocessing_agents.basic.outlier_agent import OutlierPreprocessingAgent
from agents.preprocessing_agents.basic.cate_encoding_agent import CategoryEncodingAgent
from agents.preprocessing_agents.basic.scaling_agent import ScalingAgent
from agents.preprocessing_agents.basic.duplicated_agent import DuplicatedAgent
from agents.preprocessing_agents.advanced.feature_selection_agent import FeatureSelectionAgent
from agents.preprocessing_agents.advanced.feature_engineering_agent import FeatureEngineeringAgent
from agents.preprocessing_agents.advanced.demension_agent import DimensionReductionAgent
from agents.preprocessing_agents.advanced.imbalance_agent import ImbalanceAgent

# Import KB RAG agent
from KB_rag_agents.KB_rag_agent import KnowledgeBaseRAGAgent

# Import planning agent
from planning_agents.planning_agent import PlanningAgent


@dataclass
class EDAResults:
    """EDA 결과를 저장하는 데이터클래스"""
    numeric_analysis: Dict[str, Any]
    category_analysis: Dict[str, Any]
    null_analysis: Dict[str, Any]
    correlation_analysis: Dict[str, Any]
    outlier_analysis: Dict[str, Any]
    generated_plots: List[str]  # 생성된 플롯 파일 경로들


@dataclass
class PreprocessingPlan:
    """전처리 계획을 저장하는 데이터클래스"""
    agent_name: str
    technique_names: List[str]
    priority: int
    rationale: str


@dataclass
class PreprocessingCode:
    """전처리 코드를 저장하는 데이터클래스"""
    agent_name: str
    technique_name: str
    code: str
    description: str


class PreprocessingWorkflow:
    """전처리 워크플로우 메인 클래스"""
    
    def __init__(self):
        self.eda_agents = self._initialize_eda_agents()
        self.preprocessing_agents = self._initialize_preprocessing_agents()
        self.kb_rag_agent = KnowledgeBaseRAGAgent()
        self.planning_agent = PlanningAgent()
        
    def _initialize_eda_agents(self) -> Dict[str, Any]:
        """EDA 에이전트들을 초기화"""
        return {
            'numeric_text': NumericTextAgent(),
            'numeric_image': NumericImageAgent(),
            'category_text': CategoryTextAgent(),
            'category_image': CategoryImageAgent(),
            'null_text': NullDataTextAgent(),
            'null_image': NullDataImageAgent(),
            'correlation_text': CorrelationTextAgent(),
            'correlation_image': CorrelationImageAgent(),
            'outlier_text': OutlierTextAgent(),
            'outlier_image': OutlierImageAgent()
        }
    
    def _initialize_preprocessing_agents(self) -> Dict[str, Any]:
        """전처리 에이전트들을 초기화"""
        return {
            'nulldata': NullDataPreprocessingAgent(),
            'outlier': OutlierPreprocessingAgent(),
            'category_encoding': CategoryEncodingAgent(),
            'scaling': ScalingAgent(),
            'duplicated': DuplicatedAgent(),
            'feature_selection': FeatureSelectionAgent(),
            'feature_engineering': FeatureEngineeringAgent(),
            'dimension_reduction': DimensionReductionAgent(),
            'imbalance': ImbalanceAgent()
        }
    
    def run_complete_workflow(self, df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, str]:
        """
        완전한 전처리 워크플로우 실행
        
        Args:
            df: 입력 데이터프레임
            target_column: 타겟 컬럼명 (있는 경우)
            
        Returns:
            Tuple[pd.DataFrame, str]: (전처리된 데이터프레임, 전처리 코드)
        """
        print("=== 전처리 워크플로우 시작 ===")
        
        # Step 1: DataFrame 입력 및 기본 정보 출력
        print(f"\n1. 데이터프레임 정보:")
        print(f"   - Shape: {df.shape}")
        print(f"   - Columns: {list(df.columns)}")
        print(f"   - Target column: {target_column}")
        
        # Step 2: EDA 진행 (text + image)
        print("\n2. EDA 분석 진행...")
        eda_results = self.perform_eda_analysis(df)
        
        # Step 3: EDA 정보를 기반으로 전처리 계획 수립
        print("\n3. 전처리 계획 수립...")
        preprocessing_plans = self.create_preprocessing_plans(df, eda_results, target_column)
        
        # Step 4: 각 계획에 대해 Knowledge Base에서 적절한 기법 선택
        print("\n4. Knowledge Base에서 전처리 기법 선택...")
        selected_techniques = self.select_preprocessing_techniques(preprocessing_plans, eda_results)
        
        # Step 5: LLM Coding Agent가 실제 전처리 코드 작성
        print("\n5. 전처리 코드 생성...")
        preprocessing_codes = self.generate_preprocessing_codes(selected_techniques, df, eda_results)
        
        # Step 6: 여러 전처리 코드를 통합하여 최종 전처리 실행
        print("\n6. 전처리 코드 통합 및 실행...")
        final_df, integrated_code = self.integrate_and_execute_preprocessing(df, preprocessing_codes)
        
        print("\n=== 전처리 워크플로우 완료 ===")
        return final_df, integrated_code
    
    def perform_eda_analysis(self, df: pd.DataFrame) -> EDAResults:
        """Step 2: EDA 분석 수행"""
        numeric_analysis = {}
        category_analysis = {}
        null_analysis = {}
        correlation_analysis = {}
        outlier_analysis = {}
        generated_plots = []
        
        # 수치형 변수 분석
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            print("   - 수치형 변수 분석 중...")
            numeric_analysis['text'] = self.eda_agents['numeric_text'].analyze(df, numeric_cols)
            numeric_plots = self.eda_agents['numeric_image'].generate_plots(df, numeric_cols)
            numeric_analysis['plots'] = numeric_plots
            generated_plots.extend(numeric_plots)
        
        # 범주형 변수 분석
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            print("   - 범주형 변수 분석 중...")
            category_analysis['text'] = self.eda_agents['category_text'].analyze(df, categorical_cols)
            category_plots = self.eda_agents['category_image'].generate_plots(df, categorical_cols)
            category_analysis['plots'] = category_plots
            generated_plots.extend(category_plots)
        
        # 결측값 분석
        if df.isnull().sum().sum() > 0:
            print("   - 결측값 분석 중...")
            null_analysis['text'] = self.eda_agents['null_text'].analyze(df)
            null_plots = self.eda_agents['null_image'].generate_plots(df)
            null_analysis['plots'] = null_plots
            generated_plots.extend(null_plots)
        
        # 상관관계 분석
        if len(numeric_cols) > 1:
            print("   - 상관관계 분석 중...")
            correlation_analysis['text'] = self.eda_agents['correlation_text'].analyze(df, numeric_cols)
            corr_plots = self.eda_agents['correlation_image'].generate_plots(df, numeric_cols)
            correlation_analysis['plots'] = corr_plots
            generated_plots.extend(corr_plots)
        
        # 이상치 분석
        if numeric_cols:
            print("   - 이상치 분석 중...")
            outlier_analysis['text'] = self.eda_agents['outlier_text'].analyze(df, numeric_cols)
            outlier_plots = self.eda_agents['outlier_image'].generate_plots(df, numeric_cols)
            outlier_analysis['plots'] = outlier_plots
            generated_plots.extend(outlier_plots)
        
        return EDAResults(
            numeric_analysis=numeric_analysis,
            category_analysis=category_analysis,
            null_analysis=null_analysis,
            correlation_analysis=correlation_analysis,
            outlier_analysis=outlier_analysis,
            generated_plots=generated_plots
        )
    
    def create_preprocessing_plans(self, df: pd.DataFrame, eda_results: EDAResults, 
                                 target_column: str = None) -> List[PreprocessingPlan]:
        """Step 3: EDA 정보를 기반으로 전처리 계획 수립"""
        plans = []
        
        # Planning Agent를 사용하여 전처리 계획 수립
        plan_result = self.planning_agent.create_preprocessing_plan(df, eda_results, target_column)
        
        # 계획을 PreprocessingPlan 객체로 변환
        for plan in plan_result.get('plans', []):
            plans.append(PreprocessingPlan(
                agent_name=plan['agent'],
                technique_names=plan['techniques'],
                priority=plan['priority'],
                rationale=plan['rationale']
            ))
        
        # 우선순위별로 정렬
        plans.sort(key=lambda x: x.priority)
        
        print(f"   - 총 {len(plans)}개의 전처리 계획 수립됨")
        for plan in plans:
            print(f"     * {plan.agent_name}: {plan.technique_names} (우선순위: {plan.priority})")
        
        return plans
    
    def select_preprocessing_techniques(self, plans: List[PreprocessingPlan], 
                                     eda_results: EDAResults) -> Dict[str, List[str]]:
        """Step 4: Knowledge Base에서 적절한 전처리 기법 선택"""
        selected_techniques = {}
        
        for plan in plans:
            print(f"   - {plan.agent_name} 에이전트를 위한 기법 선택 중...")
            
            # KB RAG Agent를 사용하여 적절한 기법 검색
            query = f"EDA 분석 결과: {plan.rationale}. 추천하는 전처리 기법은?"
            kb_response = self.kb_rag_agent.search_techniques(query, plan.technique_names)
            
            # 응답에서 기법 이름들 추출
            if isinstance(kb_response, dict) and 'techniques' in kb_response:
                selected_techniques[plan.agent_name] = kb_response['techniques']
            else:
                # 기본값으로 계획에서 제안된 기법들 사용
                selected_techniques[plan.agent_name] = plan.technique_names
            
            print(f"     선택된 기법: {selected_techniques[plan.agent_name]}")
        
        return selected_techniques
    
    def generate_preprocessing_codes(self, selected_techniques: Dict[str, List[str]], 
                                   df: pd.DataFrame, eda_results: EDAResults) -> List[PreprocessingCode]:
        """Step 5: LLM Coding Agent가 실제 전처리 코드 작성"""
        preprocessing_codes = []
        
        for agent_name, techniques in selected_techniques.items():
            print(f"   - {agent_name} 에이전트 코드 생성 중...")
            
            if agent_name in self.preprocessing_agents:
                agent = self.preprocessing_agents[agent_name]
                
                for technique in techniques:
                    # Knowledge Base에서 기법 정보 가져오기
                    technique_info = self.kb_rag_agent.get_technique_info(technique)
                    
                    # 에이전트를 사용하여 실제 코드 생성
                    code_result = agent.generate_code(df, technique, technique_info, eda_results)
                    
                    if code_result:
                        preprocessing_codes.append(PreprocessingCode(
                            agent_name=agent_name,
                            technique_name=technique,
                            code=code_result['code'],
                            description=code_result['description']
                        ))
                        print(f"     * {technique} 코드 생성 완료")
        
        return preprocessing_codes
    
    def integrate_and_execute_preprocessing(self, df: pd.DataFrame, 
                                          preprocessing_codes: List[PreprocessingCode]) -> Tuple[pd.DataFrame, str]:
        """Step 6: 여러 전처리 코드를 통합하여 최종 전처리 실행"""
        integrated_code_parts = [
            "# 통합 전처리 코드",
            "import pandas as pd",
            "import numpy as np",
            "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler",
            "from sklearn.impute import SimpleImputer, KNNImputer",
            "from sklearn.ensemble import IsolationForest",
            "from scipy import stats",
            "",
            "def integrated_preprocessing(df):",
            '    """통합 전처리 함수"""',
            "    df = df.copy()",
            ""
        ]
        
        # 각 전처리 코드를 통합
        current_df = df.copy()
        
        for i, preprocessing_code in enumerate(preprocessing_codes, 1):
            print(f"   - {preprocessing_code.technique_name} 실행 중...")
            
            # 코드 주석 추가
            integrated_code_parts.extend([
                f"    # {i}. {preprocessing_code.technique_name}",
                f"    # {preprocessing_code.description}",
            ])
            
            # 실제 코드 실행
            try:
                # 여기서는 실제 실행 대신 코드만 통합
                # 실제 환경에서는 exec()이나 별도의 실행 엔진 사용
                code_lines = preprocessing_code.code.split('\n')
                for line in code_lines:
                    if line.strip():
                        integrated_code_parts.append(f"    {line}")
                
                integrated_code_parts.append("")
                
                print(f"     * {preprocessing_code.technique_name} 완료")
                
            except Exception as e:
                print(f"     * {preprocessing_code.technique_name} 실행 중 오류: {e}")
                integrated_code_parts.append(f"    # ERROR: {e}")
                integrated_code_parts.append("")
        
        integrated_code_parts.extend([
            "    return df",
            "",
            "# 사용 예시:",
            "# processed_df = integrated_preprocessing(your_dataframe)"
        ])
        
        integrated_code = '\n'.join(integrated_code_parts)
        
        print(f"   - 총 {len(preprocessing_codes)}개의 전처리 단계 통합 완료")
        
        return current_df, integrated_code


def main():
    """메인 실행 함수"""
    # 샘플 데이터 생성
    np.random.seed(42)
    
    data = {
        'age': [25, 30, np.nan, 35, 40, np.nan, 45, 50, 55, 60],
        'salary': [50000, 60000, 55000, np.nan, 70000, 65000, np.nan, 80000, 85000, 90000],
        'department': ['IT', 'HR', 'IT', 'Finance', np.nan, 'IT', 'HR', 'Finance', 'IT', 'HR'],
        'experience': [2, 5, 3, 8, 10, 4, 12, 15, 18, 20],
        'target': [0, 1, 0, 1, 1, 0, 1, 1, 1, 0]
    }
    
    df = pd.DataFrame(data)
    
    print("원본 데이터:")
    print(df)
    print(f"결측값 개수: {df.isnull().sum().sum()}")
    
    # 워크플로우 실행
    workflow = PreprocessingWorkflow()
    processed_df, preprocessing_code = workflow.run_complete_workflow(df, target_column='target')
    
    print("\n=== 최종 결과 ===")
    print("생성된 전처리 코드:")
    print("=" * 50)
    print(preprocessing_code)
    print("=" * 50)


if __name__ == "__main__":
    main() 