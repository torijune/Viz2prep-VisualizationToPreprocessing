#!/usr/bin/env python3
"""
Planning Agent

EDA 결과를 분석하여 전처리 계획을 수립하는 에이전트
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class EDAResults:
    """EDA 결과를 위한 데이터클래스"""
    numeric_analysis: Dict[str, Any]
    category_analysis: Dict[str, Any]
    null_analysis: Dict[str, Any]
    correlation_analysis: Dict[str, Any]
    outlier_analysis: Dict[str, Any]
    generated_plots: List[str]


class PlanningAgent:
    """전처리 계획 수립 에이전트"""
    
    def __init__(self):
        # 에이전트별 담당 작업 정의
        self.agent_responsibilities = {
            'nulldata': {
                'tasks': ['missing_values', 'imputation'],
                'keywords': ['결측값', 'missing', 'null', 'nan', 'imputation']
            },
            'outlier': {
                'tasks': ['outlier_detection', 'outlier_removal'],
                'keywords': ['이상치', 'outlier', 'anomaly', 'extreme']
            },
            'category_encoding': {
                'tasks': ['categorical_encoding', 'label_encoding', 'onehot_encoding'],
                'keywords': ['범주형', 'categorical', 'encoding', 'label', 'onehot']
            },
            'scaling': {
                'tasks': ['normalization', 'standardization', 'scaling'],
                'keywords': ['스케일링', 'scaling', 'normalization', 'standardization']
            },
            'duplicated': {
                'tasks': ['duplicate_removal'],
                'keywords': ['중복', 'duplicate', 'redundant']
            },
            'feature_selection': {
                'tasks': ['feature_selection', 'dimensionality_reduction'],
                'keywords': ['특성선택', 'feature_selection', 'dimension_reduction']
            },
            'feature_engineering': {
                'tasks': ['feature_creation', 'feature_transformation'],
                'keywords': ['특성생성', 'feature_engineering', 'feature_creation']
            },
            'dimension_reduction': {
                'tasks': ['pca', 'dimensionality_reduction'],
                'keywords': ['차원축소', 'pca', 'dimension_reduction', 'tsne']
            },
            'imbalance': {
                'tasks': ['class_imbalance', 'oversampling', 'undersampling'],
                'keywords': ['불균형', 'imbalance', 'oversampling', 'undersampling']
            }
        }
    
    def create_preprocessing_plan(self, df: pd.DataFrame, eda_results: EDAResults, 
                                target_column: str = None) -> Dict[str, Any]:
        """
        EDA 결과를 기반으로 전처리 계획 수립
        
        Args:
            df: 입력 데이터프레임
            eda_results: EDA 분석 결과
            target_column: 타겟 컬럼명
            
        Returns:
            Dict: 전처리 계획
        """
        plans = []
        
        # 1. 결측값 처리 계획
        missing_plan = self._plan_missing_values(df, eda_results)
        if missing_plan:
            plans.append(missing_plan)
        
        # 2. 중복값 처리 계획
        duplicate_plan = self._plan_duplicate_removal(df)
        if duplicate_plan:
            plans.append(duplicate_plan)
        
        # 3. 이상치 처리 계획
        outlier_plan = self._plan_outlier_handling(df, eda_results)
        if outlier_plan:
            plans.append(outlier_plan)
        
        # 4. 범주형 인코딩 계획
        encoding_plan = self._plan_categorical_encoding(df, eda_results)
        if encoding_plan:
            plans.append(encoding_plan)
        
        # 5. 스케일링 계획
        scaling_plan = self._plan_scaling(df, eda_results)
        if scaling_plan:
            plans.append(scaling_plan)
        
        # 6. 특성 선택 계획
        feature_selection_plan = self._plan_feature_selection(df, eda_results)
        if feature_selection_plan:
            plans.append(feature_selection_plan)
        
        # 7. 특성 엔지니어링 계획
        feature_engineering_plan = self._plan_feature_engineering(df, eda_results)
        if feature_engineering_plan:
            plans.append(feature_engineering_plan)
        
        # 8. 차원 축소 계획
        dimension_reduction_plan = self._plan_dimension_reduction(df, eda_results)
        if dimension_reduction_plan:
            plans.append(dimension_reduction_plan)
        
        # 9. 클래스 불균형 처리 계획
        if target_column:
            imbalance_plan = self._plan_class_imbalance(df, target_column)
            if imbalance_plan:
                plans.append(imbalance_plan)
        
        return {
            'plans': plans,
            'total_steps': len(plans),
            'estimated_time': self._estimate_processing_time(plans),
            'data_shape': df.shape,
            'target_column': target_column
        }
    
    def _plan_missing_values(self, df: pd.DataFrame, eda_results: EDAResults) -> Optional[Dict[str, Any]]:
        """결측값 처리 계획 수립"""
        missing_count = df.isnull().sum().sum()
        if missing_count == 0:
            return None
        
        missing_ratio = missing_count / (df.shape[0] * df.shape[1])
        
        # 결측값 패턴 분석
        techniques = []
        rationale = f"총 {missing_count}개의 결측값 발견 (전체 데이터의 {missing_ratio:.2%})"
        
        # 컬럼별 결측값 비율 확인
        missing_by_column = df.isnull().sum() / len(df)
        high_missing_columns = missing_by_column[missing_by_column > 0.5]
        
        if len(high_missing_columns) > 0:
            techniques.append('drop_high_missing_columns')
            rationale += f". {len(high_missing_columns)}개 컬럼에서 50% 이상 결측값"
        
        # 수치형 변수 결측값 처리
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_missing = df[numeric_cols].isnull().sum().sum()
        if numeric_missing > 0:
            techniques.extend(['fill_numerical_median', 'advanced_imputation'])
            rationale += f". 수치형 변수에서 {numeric_missing}개 결측값"
        
        # 범주형 변수 결측값 처리
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_missing = df[categorical_cols].isnull().sum().sum()
        if categorical_missing > 0:
            techniques.append('fill_categorical_mode')
            rationale += f". 범주형 변수에서 {categorical_missing}개 결측값"
        
        return {
            'agent': 'nulldata',
            'techniques': techniques,
            'priority': 1,  # 가장 높은 우선순위
            'rationale': rationale,
            'estimated_impact': 'high'
        }
    
    def _plan_duplicate_removal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """중복값 제거 계획 수립"""
        duplicate_count = df.duplicated().sum()
        if duplicate_count == 0:
            return None
        
        duplicate_ratio = duplicate_count / len(df)
        
        return {
            'agent': 'duplicated',
            'techniques': ['remove_duplicates'],
            'priority': 2,
            'rationale': f"{duplicate_count}개의 중복 행 발견 (전체의 {duplicate_ratio:.2%})",
            'estimated_impact': 'medium'
        }
    
    def _plan_outlier_handling(self, df: pd.DataFrame, eda_results: EDAResults) -> Optional[Dict[str, Any]]:
        """이상치 처리 계획 수립"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return None
        
        # 간단한 이상치 탐지 (IQR 방법)
        outlier_counts = {}
        total_outliers = 0
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                outlier_counts[col] = outliers
                total_outliers += outliers
        
        if total_outliers == 0:
            return None
        
        techniques = ['iqr_outlier_detection']
        rationale = f"{total_outliers}개의 이상치 발견"
        
        # 이상치 비율에 따라 처리 방법 결정
        outlier_ratio = total_outliers / (len(df) * len(numeric_cols))
        if outlier_ratio > 0.1:
            techniques.extend(['zscore_outlier_detection', 'isolation_forest_outliers'])
            rationale += " (높은 비율로 인해 다중 방법 적용 권장)"
        
        return {
            'agent': 'outlier',
            'techniques': techniques,
            'priority': 3,
            'rationale': rationale,
            'estimated_impact': 'medium'
        }
    
    def _plan_categorical_encoding(self, df: pd.DataFrame, eda_results: EDAResults) -> Optional[Dict[str, Any]]:
        """범주형 인코딩 계획 수립"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) == 0:
            return None
        
        techniques = []
        rationale = f"{len(categorical_cols)}개의 범주형 변수 발견"
        
        # 카디널리티에 따른 인코딩 전략
        low_cardinality_cols = []
        medium_cardinality_cols = []
        high_cardinality_cols = []
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count <= 10:
                low_cardinality_cols.append(col)
            elif unique_count <= 20:
                medium_cardinality_cols.append(col)
            else:
                high_cardinality_cols.append(col)
        
        if low_cardinality_cols:
            techniques.append('label_encoding')
            rationale += f". {len(low_cardinality_cols)}개 저카디널리티"
        
        if medium_cardinality_cols:
            techniques.append('onehot_encoding')
            rationale += f". {len(medium_cardinality_cols)}개 중카디널리티"
        
        if high_cardinality_cols:
            techniques.append('frequency_encoding')
            rationale += f". {len(high_cardinality_cols)}개 고카디널리티"
        
        return {
            'agent': 'category_encoding',
            'techniques': techniques,
            'priority': 4,
            'rationale': rationale,
            'estimated_impact': 'high'
        }
    
    def _plan_scaling(self, df: pd.DataFrame, eda_results: EDAResults) -> Optional[Dict[str, Any]]:
        """스케일링 계획 수립"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) <= 1:
            return None
        
        # 변수들의 스케일 차이 확인
        scales = df[numeric_cols].std()
        max_scale = scales.max()
        min_scale = scales.min()
        
        if max_scale / min_scale < 10:  # 스케일 차이가 크지 않음
            return None
        
        techniques = ['standard_scaling']
        rationale = f"{len(numeric_cols)}개 수치형 변수의 스케일 차이 발견 (최대/최소 비율: {max_scale/min_scale:.1f})"
        
        # 이상치가 많으면 Robust Scaling 추천
        if 'outlier' in str(eda_results.outlier_analysis):
            techniques.append('robust_scaling')
            rationale += ". 이상치로 인해 Robust Scaling 추가 권장"
        
        # 범위 제한이 필요한 경우 MinMax Scaling 추천
        techniques.append('minmax_scaling')
        
        return {
            'agent': 'scaling',
            'techniques': techniques,
            'priority': 5,
            'rationale': rationale,
            'estimated_impact': 'medium'
        }
    
    def _plan_feature_selection(self, df: pd.DataFrame, eda_results: EDAResults) -> Optional[Dict[str, Any]]:
        """특성 선택 계획 수립"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # 변수가 너무 많은 경우에만 특성 선택 수행
        if len(df.columns) < 20:
            return None
        
        techniques = ['variance_threshold']
        rationale = f"총 {len(df.columns)}개 변수로 특성 선택 필요"
        
        if len(numeric_cols) > 5:
            techniques.append('correlation_filter')
            rationale += ". 다중공선성 제거 권장"
        
        return {
            'agent': 'feature_selection',
            'techniques': techniques,
            'priority': 6,
            'rationale': rationale,
            'estimated_impact': 'low'
        }
    
    def _plan_feature_engineering(self, df: pd.DataFrame, eda_results: EDAResults) -> Optional[Dict[str, Any]]:
        """특성 엔지니어링 계획 수립"""
        techniques = []
        rationale = "특성 엔지니어링 기회 탐색"
        
        # 날짜/시간 컬럼 확인
        datetime_candidates = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # 샘플 값들을 확인하여 날짜 형식인지 판단
                sample_values = df[col].dropna().head(10).astype(str)
                if any('/' in str(val) or '-' in str(val) for val in sample_values):
                    datetime_candidates.append(col)
        
        if datetime_candidates:
            techniques.append('datetime_features')
            rationale += f". {len(datetime_candidates)}개 날짜 컬럼 발견"
        
        # 수치형 변수가 충분한 경우 상호작용 특성 생성
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 3:
            techniques.append('polynomial_features')
            rationale += f". {len(numeric_cols)}개 수치형 변수로 상호작용 특성 생성 가능"
        
        # 연속형 변수의 구간화
        if len(numeric_cols) > 0:
            techniques.append('binning_features')
            rationale += ". 수치형 변수 구간화 적용"
        
        if not techniques:
            return None
        
        return {
            'agent': 'feature_engineering',
            'techniques': techniques,
            'priority': 7,
            'rationale': rationale,
            'estimated_impact': 'medium'
        }
    
    def _plan_dimension_reduction(self, df: pd.DataFrame, eda_results: EDAResults) -> Optional[Dict[str, Any]]:
        """차원 축소 계획 수립"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # 변수가 충분히 많고 데이터가 충분한 경우에만 수행
        if len(numeric_cols) < 10 or len(df) < 100:
            return None
        
        techniques = ['pca_reduction']
        rationale = f"{len(numeric_cols)}개 수치형 변수로 차원 축소 고려"
        
        # 데이터 크기에 따라 추가 기법 결정
        if len(df) > 1000:
            techniques.extend(['tsne_reduction', 'umap_reduction'])
            rationale += ". 충분한 데이터로 비선형 차원축소 적용 가능"
        
        return {
            'agent': 'dimension_reduction',
            'techniques': techniques,
            'priority': 8,
            'rationale': rationale,
            'estimated_impact': 'low'
        }
    
    def _plan_class_imbalance(self, df: pd.DataFrame, target_column: str) -> Optional[Dict[str, Any]]:
        """클래스 불균형 처리 계획 수립"""
        if target_column not in df.columns:
            return None
        
        # 타겟 변수의 분포 확인
        target_counts = df[target_column].value_counts()
        if len(target_counts) < 2:
            return None
        
        # 불균형 비율 계산
        majority_class_ratio = target_counts.max() / len(df)
        minority_class_ratio = target_counts.min() / len(df)
        imbalance_ratio = majority_class_ratio / minority_class_ratio
        
        # 불균형이 심하지 않으면 처리하지 않음
        if imbalance_ratio < 2:
            return None
        
        techniques = ['class_weights']
        rationale = f"클래스 불균형 발견 (비율: {imbalance_ratio:.1f}:1)"
        
        # 불균형 정도에 따라 추가 기법 결정
        if imbalance_ratio > 5:
            techniques.extend(['smote_oversampling', 'random_undersampling'])
            rationale += ". 심한 불균형으로 리샘플링 권장"
        
        return {
            'agent': 'imbalance',
            'techniques': techniques,
            'priority': 9,
            'rationale': rationale,
            'estimated_impact': 'high'
        }
    
    def _estimate_processing_time(self, plans: List[Dict[str, Any]]) -> str:
        """처리 시간 추정"""
        total_steps = len(plans)
        if total_steps <= 3:
            return "빠름 (< 1분)"
        elif total_steps <= 6:
            return "보통 (1-3분)"
        else:
            return "느림 (> 3분)"


def main():
    """테스트 함수"""
    # 샘플 데이터 생성
    np.random.seed(42)
    
    data = {
        'age': [25, 30, np.nan, 35, 40, np.nan, 45, 50, 55, 60] * 10,
        'salary': [50000, 60000, 55000, np.nan, 70000, 65000, np.nan, 80000, 85000, 90000] * 10,
        'department': ['IT', 'HR', 'IT', 'Finance', np.nan, 'IT', 'HR', 'Finance', 'IT', 'HR'] * 10,
        'experience': list(range(100)),
        'target': [0, 1, 0, 1, 1, 0, 1, 1, 1, 0] * 10
    }
    
    df = pd.DataFrame(data)
    
    # 임시 EDA 결과 생성
    eda_results = EDAResults(
        numeric_analysis={'summary': 'numeric analysis done'},
        category_analysis={'summary': 'category analysis done'},
        null_analysis={'summary': 'null analysis done'},
        correlation_analysis={'summary': 'correlation analysis done'},
        outlier_analysis={'summary': 'outlier analysis done'},
        generated_plots=['plot1.png', 'plot2.png']
    )
    
    # Planning Agent 테스트
    planning_agent = PlanningAgent()
    plan = planning_agent.create_preprocessing_plan(df, eda_results, target_column='target')
    
    print("=== 전처리 계획 ===")
    print(f"총 {plan['total_steps']}개 단계")
    print(f"예상 처리 시간: {plan['estimated_time']}")
    print(f"데이터 형태: {plan['data_shape']}")
    print()
    
    for plan_item in plan['plans']:
        print(f"에이전트: {plan_item['agent']}")
        print(f"우선순위: {plan_item['priority']}")
        print(f"기법들: {plan_item['techniques']}")
        print(f"근거: {plan_item['rationale']}")
        print(f"예상 영향: {plan_item['estimated_impact']}")
        print("-" * 50)


if __name__ == "__main__":
    main()