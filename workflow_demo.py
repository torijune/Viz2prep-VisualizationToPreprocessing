#!/usr/bin/env python3
"""
전처리 워크플로우 데모

Flow:
1. DataFrame 입력 (시뮬레이션)
2. EDA 진행 (text + image) (시뮬레이션)
3. EDA 정보를 각 preprocessing agents에게 전달
4. MultiModal LLM이 Knowledge Base에서 적절한 전처리 방법 선택
5. LLM Coding Agent가 실제 전처리 코드 작성
6. 여러 전처리 코드를 통합하여 최종 전처리 실행
"""

import json
import sys
import os
from typing import Dict, List, Any

# Knowledge Base 함수들 import
sys.path.append(os.path.join(os.path.dirname(__file__), 'KB_rag_agents', 'knowledge_base'))
from KB_rag_agents.KB_rag_agent import KnowledgeBaseRAGAgent


class WorkflowDemo:
    """전처리 워크플로우 데모 클래스"""
    
    def __init__(self):
        self.kb_agent = KnowledgeBaseRAGAgent()
        
    def simulate_dataframe_info(self):
        """Step 1: DataFrame 정보 시뮬레이션"""
        return {
            'shape': (1000, 10),
            'columns': ['age', 'salary', 'department', 'experience', 'education', 
                       'performance_score', 'years_in_company', 'location', 'gender', 'target'],
            'numeric_columns': ['age', 'salary', 'experience', 'performance_score', 'years_in_company'],
            'categorical_columns': ['department', 'education', 'location', 'gender'],
            'target_column': 'target',
            'missing_values': {'age': 45, 'salary': 23, 'department': 12},
            'data_types': {
                'age': 'numeric', 'salary': 'numeric', 'department': 'categorical',
                'experience': 'numeric', 'education': 'categorical'
            }
        }
    
    def simulate_eda_analysis(self, df_info):
        """Step 2: EDA 분석 시뮬레이션"""
        return {
            'numeric_analysis': {
                'summary': f"{len(df_info['numeric_columns'])}개의 수치형 변수 분석 완료",
                'findings': [
                    "salary 변수에서 이상치 발견",
                    "age와 experience 간 높은 상관관계",
                    "performance_score 분포가 정규분포에 가까움"
                ]
            },
            'categorical_analysis': {
                'summary': f"{len(df_info['categorical_columns'])}개의 범주형 변수 분석 완료",
                'findings': [
                    "department 변수에 10개의 고유값",
                    "education 변수에 5개의 고유값",
                    "location 변수에 25개의 고유값 (고카디널리티)"
                ]
            },
            'null_analysis': {
                'summary': f"총 {sum(df_info['missing_values'].values())}개의 결측값 발견",
                'findings': [
                    f"age: {df_info['missing_values']['age']}개 결측값",
                    f"salary: {df_info['missing_values']['salary']}개 결측값",
                    f"department: {df_info['missing_values']['department']}개 결측값"
                ]
            },
            'outlier_analysis': {
                'summary': "이상치 분석 완료",
                'findings': [
                    "salary 변수에서 극단값 발견",
                    "performance_score에서 일부 이상치 존재"
                ]
            },
            'correlation_analysis': {
                'summary': "상관관계 분석 완료",
                'findings': [
                    "age와 experience 강한 양의 상관관계 (0.85)",
                    "salary와 performance_score 중간 상관관계 (0.6)"
                ]
            }
        }
    
    def create_preprocessing_plans(self, df_info, eda_results):
        """Step 3: 전처리 계획 수립"""
        plans = []
        
        # 결측값 처리 계획
        if any(df_info['missing_values'].values()):
            plans.append({
                'agent': 'nulldata',
                'techniques': ['fill_numerical_median', 'fill_categorical_mode'],
                'priority': 1,
                'rationale': f"총 {sum(df_info['missing_values'].values())}개의 결측값 처리 필요"
            })
        
        # 이상치 처리 계획
        if "이상치" in str(eda_results['outlier_analysis']):
            plans.append({
                'agent': 'outlier',
                'techniques': ['iqr_outlier_detection', 'zscore_outlier_detection'],
                'priority': 2,
                'rationale': "salary와 performance_score에서 이상치 발견"
            })
        
        # 범주형 인코딩 계획
        if df_info['categorical_columns']:
            plans.append({
                'agent': 'category_encoding',
                'techniques': ['label_encoding', 'onehot_encoding', 'frequency_encoding'],
                'priority': 3,
                'rationale': f"{len(df_info['categorical_columns'])}개의 범주형 변수 인코딩 필요"
            })
        
        # 스케일링 계획
        if len(df_info['numeric_columns']) > 1:
            plans.append({
                'agent': 'scaling',
                'techniques': ['standard_scaling', 'minmax_scaling'],
                'priority': 4,
                'rationale': "수치형 변수들의 스케일 차이로 인한 정규화 필요"
            })
        
        # 특성 선택 계획
        if len(df_info['columns']) > 8:
            plans.append({
                'agent': 'feature_selection',
                'techniques': ['correlation_filter', 'variance_threshold'],
                'priority': 5,
                'rationale': "높은 상관관계를 가진 변수들로 인한 특성 선택 필요"
            })
        
        return plans
    
    def select_preprocessing_techniques(self, plans, eda_results):
        """Step 4: Knowledge Base에서 전처리 기법 선택"""
        selected_techniques = {}
        
        for plan in plans:
            print(f"   - {plan['agent']} 에이전트를 위한 기법 선택 중...")
            
            # KB RAG Agent를 사용하여 적절한 기법 검색
            query = f"EDA 분석 결과: {plan['rationale']}. 추천하는 전처리 기법은?"
            kb_response = self.kb_agent.search_techniques(query, plan['techniques'])
            
            selected_techniques[plan['agent']] = kb_response['techniques']
            print(f"     선택된 기법: {selected_techniques[plan['agent']]}")
        
        return selected_techniques
    
    def generate_preprocessing_codes(self, selected_techniques, df_info, eda_results):
        """Step 5: 전처리 코드 생성"""
        preprocessing_codes = []
        
        for agent_name, techniques in selected_techniques.items():
            print(f"   - {agent_name} 에이전트 코드 생성 중...")
            
            for technique in techniques:
                # Knowledge Base에서 기법 정보 가져오기
                technique_info = self.kb_agent.get_technique_info(technique)
                
                if technique_info:
                    preprocessing_codes.append({
                        'agent_name': agent_name,
                        'technique_name': technique,
                        'code': technique_info['code'],
                        'description': technique_info['description']
                    })
                    print(f"     * {technique} 코드 생성 완료")
        
        return preprocessing_codes
    
    def integrate_preprocessing_codes(self, preprocessing_codes):
        """Step 6: 전처리 코드 통합"""
        integrated_code_parts = [
            "# 통합 전처리 코드",
            "# 자동 생성된 전처리 파이프라인",
            "",
            "import pandas as pd",
            "import numpy as np",
            "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder",
            "from sklearn.impute import SimpleImputer, KNNImputer",
            "from sklearn.ensemble import IsolationForest",
            "from scipy import stats",
            "",
            "def integrated_preprocessing(df):",
            '    """',
            '    통합 전처리 함수',
            '    자동으로 생성된 전처리 파이프라인',
            '    """',
            "    df = df.copy()",
            "    print('전처리 시작...')",
            ""
        ]
        
        for i, code_info in enumerate(preprocessing_codes, 1):
            print(f"   - {code_info['technique_name']} 통합 중...")
            
            # 코드 주석 추가
            integrated_code_parts.extend([
                f"    # Step {i}: {code_info['technique_name']}",
                f"    # {code_info['description']}",
                f"    print('Step {i}: {code_info['technique_name']} 실행 중...')",
            ])
            
            # 실제 코드 추가 (들여쓰기 조정)
            code_lines = code_info['code'].split('\n')
            for line in code_lines:
                if line.strip():
                    # 함수 정의 라인은 주석으로 처리하고 내용만 추가
                    if line.strip().startswith('def '):
                        integrated_code_parts.append(f"    # {line.strip()}")
                    elif line.strip().startswith('import ') or line.strip().startswith('from '):
                        # import 문은 이미 위에 있으므로 주석으로
                        integrated_code_parts.append(f"    # {line.strip()}")
                    else:
                        # 들여쓰기 조정
                        adjusted_line = "    " + line.strip()
                        integrated_code_parts.append(adjusted_line)
            
            integrated_code_parts.extend([
                f"    print('Step {i}: {code_info['technique_name']} 완료')",
                ""
            ])
        
        integrated_code_parts.extend([
            "    print('전처리 완료!')",
            "    return df",
            "",
            "# 사용 예시:",
            "# processed_df = integrated_preprocessing(your_dataframe)",
            "# print('전처리된 데이터 형태:', processed_df.shape)"
        ])
        
        return '\n'.join(integrated_code_parts)
    
    def run_complete_workflow(self):
        """완전한 워크플로우 실행"""
        print("=== 전처리 워크플로우 시작 ===")
        
        # Step 1: DataFrame 정보
        print("\n1. 데이터프레임 정보 분석...")
        df_info = self.simulate_dataframe_info()
        print(f"   - 데이터 형태: {df_info['shape']}")
        print(f"   - 총 컬럼 수: {len(df_info['columns'])}")
        print(f"   - 수치형 변수: {len(df_info['numeric_columns'])}개")
        print(f"   - 범주형 변수: {len(df_info['categorical_columns'])}개")
        print(f"   - 결측값: {sum(df_info['missing_values'].values())}개")
        
        # Step 2: EDA 분석
        print("\n2. EDA 분석 진행...")
        eda_results = self.simulate_eda_analysis(df_info)
        for analysis_type, result in eda_results.items():
            print(f"   - {analysis_type}: {result['summary']}")
        
        # Step 3: 전처리 계획 수립
        print("\n3. 전처리 계획 수립...")
        plans = self.create_preprocessing_plans(df_info, eda_results)
        print(f"   - 총 {len(plans)}개의 전처리 단계 계획됨")
        for plan in plans:
            print(f"     * {plan['agent']}: {plan['techniques']} (우선순위: {plan['priority']})")
        
        # Step 4: Knowledge Base에서 기법 선택
        print("\n4. Knowledge Base에서 전처리 기법 선택...")
        selected_techniques = self.select_preprocessing_techniques(plans, eda_results)
        
        # Step 5: 전처리 코드 생성
        print("\n5. 전처리 코드 생성...")
        preprocessing_codes = self.generate_preprocessing_codes(selected_techniques, df_info, eda_results)
        print(f"   - 총 {len(preprocessing_codes)}개의 전처리 코드 생성됨")
        
        # Step 6: 코드 통합
        print("\n6. 전처리 코드 통합...")
        integrated_code = self.integrate_preprocessing_codes(preprocessing_codes)
        print(f"   - 통합 코드 생성 완료 ({len(integrated_code.split('\\n'))} 라인)")
        
        print("\n=== 전처리 워크플로우 완료 ===")
        
        return integrated_code


def main():
    """메인 실행 함수"""
    workflow = WorkflowDemo()
    
    # 워크플로우 실행
    final_code = workflow.run_complete_workflow()
    
    print("\n" + "="*60)
    print("최종 생성된 통합 전처리 코드:")
    print("="*60)
    print(final_code)
    print("="*60)
    
    # 파일로 저장
    with open('generated_preprocessing_pipeline.py', 'w', encoding='utf-8') as f:
        f.write(final_code)
    
    print(f"\n코드가 'generated_preprocessing_pipeline.py' 파일로 저장되었습니다!")


if __name__ == "__main__":
    main() 