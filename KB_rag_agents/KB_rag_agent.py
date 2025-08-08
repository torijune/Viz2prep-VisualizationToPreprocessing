#!/usr/bin/env python3
"""
Knowledge Base RAG Agent

전처리 기법들에 대한 정보를 Knowledge Base에서 검색하고,
적절한 전처리 방법을 추천하는 RAG 에이전트
"""

import json
import os
from typing import Dict, List, Any, Optional
from difflib import SequenceMatcher
import sys

# Knowledge Base functions import
kb_path = os.path.join(os.path.dirname(__file__), 'knowledge_base')
sys.path.append(kb_path)
from preprocessing_functions import get_code_snippet, get_available_techniques, TECHNIQUE_CODE


class KnowledgeBaseRAGAgent:
    """Knowledge Base를 활용한 RAG 에이전트"""
    
    def __init__(self):
        self.kb_path = os.path.join(os.path.dirname(__file__), 'knowledge_base', 'preprocessing_codes.json')
        self.preprocessing_info = self._load_preprocessing_info()
        self.available_techniques = get_available_techniques()
        
    def _load_preprocessing_info(self) -> Dict[str, Any]:
        """Knowledge Base JSON 파일 로드"""
        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Knowledge base file not found at {self.kb_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Warning: Error parsing knowledge base JSON: {e}")
            return {}
    
    def search_techniques(self, query: str, suggested_techniques: List[str] = None) -> Dict[str, Any]:
        """
        쿼리에 기반하여 적절한 전처리 기법 검색
        
        Args:
            query: 검색 쿼리 (EDA 분석 결과나 요구사항)
            suggested_techniques: 제안된 기법들 (선택사항)
            
        Returns:
            Dict: 추천된 기법들과 관련 정보
        """
        print(f"🔍 [KB-RAG] '{query}' 쿼리로 기법 검색 중...")
        
        # 키워드 기반 매칭
        query_lower = query.lower()
        matched_techniques = []
        
        # 1. 제안된 기법들이 있으면 우선 검증
        if suggested_techniques:
            for technique in suggested_techniques:
                if technique in self.available_techniques:
                    matched_techniques.append(technique)
                    print(f"   ✅ [KB-RAG] 제안된 기법 '{technique}' 추가")
        
        # 2. 쿼리 기반으로 추가 기법 검색
        keyword_mapping = {
            '결측값': ['fill_categorical_mode', 'fill_numerical_median', 'advanced_imputation', 'drop_high_missing_columns'],
            'missing': ['fill_categorical_mode', 'fill_numerical_median', 'advanced_imputation', 'drop_high_missing_columns'],
            '이상치': ['iqr_outlier_detection', 'zscore_outlier_detection', 'isolation_forest_outliers'],
            'outlier': ['iqr_outlier_detection', 'zscore_outlier_detection', 'isolation_forest_outliers'],
            '인코딩': ['label_encoding', 'onehot_encoding', 'frequency_encoding'],
            'encoding': ['label_encoding', 'onehot_encoding', 'frequency_encoding'],
            '스케일링': ['standard_scaling', 'minmax_scaling', 'robust_scaling'],
            'scaling': ['standard_scaling', 'minmax_scaling', 'robust_scaling'],
            '정규화': ['standard_scaling', 'minmax_scaling'],
            'normalization': ['standard_scaling', 'minmax_scaling'],
            '차원축소': ['pca_reduction', 'tsne_reduction', 'umap_reduction'],
            'dimension': ['pca_reduction', 'tsne_reduction', 'umap_reduction'],
            '불균형': ['smote_oversampling', 'random_undersampling', 'class_weights'],
            'imbalance': ['smote_oversampling', 'random_undersampling', 'class_weights'],
            '특성선택': ['variance_threshold', 'correlation_filter', 'recursive_feature_elimination'],
            'feature_selection': ['variance_threshold', 'correlation_filter', 'recursive_feature_elimination'],
            '특성생성': ['polynomial_features', 'datetime_features', 'binning_features'],
            'feature_engineering': ['polynomial_features', 'datetime_features', 'binning_features']
        }
        
        for keyword, techniques in keyword_mapping.items():
            if keyword in query_lower:
                print(f"   🎯 [KB-RAG] '{keyword}' 키워드 매칭됨")
                for technique in techniques:
                    if technique not in matched_techniques:
                        matched_techniques.append(technique)
                        print(f"      ✅ [KB-RAG] '{technique}' 기법 추가")
                break
        
        # 3. 제안된 기법이 없으면 기본 기법들 추천
        if not matched_techniques:
            print("   ⚠️  [KB-RAG] 매칭되는 키워드가 없어 기본 기법 추천")
            # 기본적으로 자주 사용되는 기법들
            matched_techniques = ['fill_numerical_median', 'iqr_outlier_detection', 'standard_scaling']
        
        print(f"✅ [KB-RAG] 검색 완료 - {len(matched_techniques)}개 기법 발견")
        for technique in matched_techniques:
            print(f"   - {technique}")
        
        return {
            'query': query,
            'techniques': matched_techniques[:5],  # 최대 5개까지
            'total_available': len(self.available_techniques)
        }
    
    def get_technique_info(self, technique_name: str) -> Optional[Dict[str, Any]]:
        """
        특정 기법의 상세 정보 가져오기
        
        Args:
            technique_name: 기법 이름
            
        Returns:
            Dict: 기법 정보 (코드, 설명 등)
        """
        # Knowledge Base에서 메타데이터 검색
        technique_metadata = None
        for category, info in self.preprocessing_info.items():
            for technique in info.get('techniques', []):
                if technique['name'] == technique_name:
                    technique_metadata = technique.copy()
                    technique_metadata['category'] = category
                    break
            if technique_metadata:
                break
        
        # 코드 가져오기
        code = get_code_snippet(technique_name)
        
        if technique_metadata and code:
            return {
                'name': technique_name,
                'description': technique_metadata.get('description', ''),
                'use_case': technique_metadata.get('use_case', ''),
                'keywords': technique_metadata.get('keywords', []),
                'category': technique_metadata.get('category', ''),
                'code': code
            }
        elif code:
            return {
                'name': technique_name,
                'description': f'전처리 기법: {technique_name}',
                'use_case': '데이터 전처리',
                'keywords': [technique_name],
                'category': 'unknown',
                'code': code
            }
        else:
            return None
    
    def search_by_category(self, category: str) -> List[str]:
        """
        카테고리별로 기법 검색
        
        Args:
            category: 카테고리 이름
            
        Returns:
            List[str]: 해당 카테고리의 기법들
        """
        if category in self.preprocessing_info:
            return [tech['name'] for tech in self.preprocessing_info[category].get('techniques', [])]
        return []
    
    def search_by_keywords(self, keywords: List[str]) -> List[str]:
        """
        키워드로 기법 검색
        
        Args:
            keywords: 검색할 키워드들
            
        Returns:
            List[str]: 매칭된 기법들
        """
        matched_techniques = []
        keywords_lower = [kw.lower() for kw in keywords]
        
        for category, info in self.preprocessing_info.items():
            for technique in info.get('techniques', []):
                technique_keywords = [kw.lower() for kw in technique.get('keywords', [])]
                
                # 키워드 매칭
                for kw in keywords_lower:
                    if any(kw in tech_kw for tech_kw in technique_keywords):
                        if technique['name'] not in matched_techniques:
                            matched_techniques.append(technique['name'])
                        break
        
        return matched_techniques
    
    def get_similar_techniques(self, technique_name: str, threshold: float = 0.6) -> List[str]:
        """
        유사한 기법들 찾기
        
        Args:
            technique_name: 기준 기법 이름
            threshold: 유사도 임계값
            
        Returns:
            List[str]: 유사한 기법들
        """
        similar_techniques = []
        
        for available_tech in self.available_techniques:
            if available_tech != technique_name:
                similarity = SequenceMatcher(None, technique_name, available_tech).ratio()
                if similarity >= threshold:
                    similar_techniques.append((available_tech, similarity))
        
        # 유사도 순으로 정렬
        similar_techniques.sort(key=lambda x: x[1], reverse=True)
        
        return [tech[0] for tech in similar_techniques]
    
    def recommend_preprocessing_pipeline(self, eda_summary: str) -> Dict[str, Any]:
        """
        EDA 요약을 기반으로 전처리 파이프라인 추천
        
        Args:
            eda_summary: EDA 분석 요약
            
        Returns:
            Dict: 추천 파이프라인
        """
        pipeline_steps = []
        
        # 우선순위 기반 파이프라인 구성
        if '결측값' in eda_summary or 'missing' in eda_summary.lower():
            missing_techniques = self.search_techniques("결측값 처리 필요")['techniques']
            pipeline_steps.append({
                'step': 1,
                'category': 'missing_values',
                'techniques': missing_techniques[:2],
                'rationale': '결측값 처리가 필요함'
            })
        
        if '이상치' in eda_summary or 'outlier' in eda_summary.lower():
            outlier_techniques = self.search_techniques("이상치 처리 필요")['techniques']
            pipeline_steps.append({
                'step': 2,
                'category': 'outliers',
                'techniques': outlier_techniques[:2],
                'rationale': '이상치 처리가 필요함'
            })
        
        if '범주형' in eda_summary or 'categorical' in eda_summary.lower():
            encoding_techniques = self.search_techniques("범주형 인코딩 필요")['techniques']
            pipeline_steps.append({
                'step': 3,
                'category': 'categorical_encoding',
                'techniques': encoding_techniques[:2],
                'rationale': '범주형 변수 인코딩이 필요함'
            })
        
        if '스케일링' in eda_summary or 'scaling' in eda_summary.lower():
            scaling_techniques = self.search_techniques("스케일링 필요")['techniques']
            pipeline_steps.append({
                'step': 4,
                'category': 'scaling',
                'techniques': scaling_techniques[:2],
                'rationale': '변수 스케일링이 필요함'
            })
        
        return {
            'pipeline': pipeline_steps,
            'total_steps': len(pipeline_steps),
            'estimated_techniques': sum(len(step['techniques']) for step in pipeline_steps)
        }
    
    def get_category_info(self) -> Dict[str, Dict[str, Any]]:
        """
        모든 카테고리 정보 반환
        
        Returns:
            Dict: 카테고리별 정보
        """
        category_summary = {}
        
        for category, info in self.preprocessing_info.items():
            techniques = info.get('techniques', [])
            category_summary[category] = {
                'description': info.get('description', ''),
                'technique_count': len(techniques),
                'technique_names': [tech['name'] for tech in techniques]
            }
        
        return category_summary
    
    def validate_technique(self, technique_name: str) -> bool:
        """
        기법 이름이 유효한지 검증
        
        Args:
            technique_name: 검증할 기법 이름
            
        Returns:
            bool: 유효 여부
        """
        return technique_name in self.available_techniques


def main():
    """테스트 함수"""
    agent = KnowledgeBaseRAGAgent()
    
    print("=== Knowledge Base RAG Agent 테스트 ===")
    
    # 1. 기법 검색 테스트
    print("\n1. 결측값 관련 기법 검색:")
    result = agent.search_techniques("결측값이 많이 발견됨. 처리 방법 추천")
    print(f"추천 기법: {result['techniques']}")
    
    # 2. 특정 기법 정보 가져오기
    print("\n2. fill_numerical_median 기법 정보:")
    info = agent.get_technique_info("fill_numerical_median")
    if info:
        print(f"설명: {info['description']}")
        print(f"사용 케이스: {info['use_case']}")
        print(f"카테고리: {info['category']}")
    
    # 3. 카테고리별 검색
    print("\n3. missing_values 카테고리 기법들:")
    missing_techniques = agent.search_by_category("missing_values")
    print(f"기법들: {missing_techniques}")
    
    # 4. 전처리 파이프라인 추천
    print("\n4. 전처리 파이프라인 추천:")
    eda_summary = "결측값이 발견되고, 이상치가 있으며, 범주형 변수가 존재함"
    pipeline = agent.recommend_preprocessing_pipeline(eda_summary)
    print(f"파이프라인 단계 수: {pipeline['total_steps']}")
    for step in pipeline['pipeline']:
        print(f"  Step {step['step']}: {step['category']} - {step['techniques']}")


if __name__ == "__main__":
    main()