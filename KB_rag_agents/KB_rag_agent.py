"""
Knowledge Base RAG Agent
전처리 코드 Knowledge Base를 활용하여 RAG 기반 코드 생성을 수행합니다.
"""

import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import os

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.runnables import RunnableLambda


class KBRAGAgent:
    """
    Knowledge Base RAG Agent
    전처리 코드 Knowledge Base를 검색하고 관련 코드를 찾아 LLM 코드 생성을 지원합니다.
    """
    
    def __init__(self):
        """KB RAG Agent 초기화"""
        self.kb_path = Path(__file__).parent / "knowledge_base"
        self.kb_path.mkdir(exist_ok=True)
        
        # OpenAI 클라이언트 및 임베딩 모델 초기화
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=2000
        )
        
        # Knowledge Base 로드 또는 생성
        self.knowledge_base = self.load_or_create_kb()
        self.code_embeddings = self.load_or_create_embeddings()
        
        print("KB RAG Agent 초기화 완료")
    
    def load_or_create_kb(self) -> Dict:
        """Knowledge Base 로드 또는 생성"""
        kb_file = self.kb_path / "preprocessing_codes.json"
        
        if kb_file.exists():
            with open(kb_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print("Knowledge Base 파일이 없습니다. preprocessing_codes.json 파일을 생성해주세요.")
            return {}
    
    def load_or_create_embeddings(self) -> Dict:
        """임베딩 로드 또는 생성"""
        embeddings_file = self.kb_path / "embeddings.pkl"
        
        if embeddings_file.exists():
            with open(embeddings_file, 'rb') as f:
                return pickle.load(f)
        else:
            # 임베딩 생성
            embeddings = self.create_embeddings()
            with open(embeddings_file, 'wb') as f:
                pickle.dump(embeddings, f)
            return embeddings
    
    def create_embeddings(self) -> Dict:
        """Knowledge Base의 각 기법에 대한 임베딩 생성"""
        print("임베딩 생성 중...")
        embeddings_dict = {}
        
        for category, data in self.knowledge_base.items():
            embeddings_dict[category] = []
            
            for technique in data["techniques"]:
                # 기법 설명과 키워드를 결합하여 텍스트 생성
                text_for_embedding = f"{technique['description']} {technique['use_case']} {' '.join(technique['keywords'])}"
                
                # OpenAI 임베딩 생성
                embedding = self.embeddings.embed_query(text_for_embedding)
                
                embeddings_dict[category].append({
                    'name': technique['name'],
                    'embedding': embedding,
                    'text': text_for_embedding
                })
        
        print("임베딩 생성 완료")
        return embeddings_dict
    
    def search_relevant_codes(self, query: str, top_k: int = 5) -> List[Dict]:
        """쿼리와 관련된 코드들을 검색"""
        query_embedding = self.embeddings.embed_query(query)
        
        all_similarities = []
        
        # 모든 카테고리에서 유사도 계산
        for category, embeddings_list in self.code_embeddings.items():
            for item in embeddings_list:
                similarity = cosine_similarity(
                    [query_embedding], 
                    [item['embedding']]
                )[0][0]
                
                # Knowledge Base에서 해당 기법 정보 찾기
                technique_info = None
                for tech in self.knowledge_base[category]['techniques']:
                    if tech['name'] == item['name']:
                        technique_info = tech
                        break
                
                if technique_info:
                    all_similarities.append({
                        'category': category,
                        'technique': technique_info,
                        'similarity': similarity
                    })
        
        # 유사도 순으로 정렬하고 상위 k개 반환
        all_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return all_similarities[:top_k]
    
    def generate_code_with_rag(self, eda_results: Dict[str, Any], planning_info: Dict[str, Any]) -> str:
        """RAG를 사용하여 전처리 코드 생성"""
        
        # Planning 정보에서 검색 쿼리 생성
        search_queries = []
        if 'missing_values' in planning_info.get('tasks', []):
            search_queries.append("missing values imputation filling")
        if 'outliers' in planning_info.get('tasks', []):
            search_queries.append("outlier detection removal capping")
        if 'categorical_encoding' in planning_info.get('tasks', []):
            search_queries.append("categorical encoding label onehot")
        if 'scaling' in planning_info.get('tasks', []):
            search_queries.append("feature scaling normalization")
        if 'feature_selection' in planning_info.get('tasks', []):
            search_queries.append("feature selection variance correlation")
        if 'feature_engineering' in planning_info.get('tasks', []):
            search_queries.append("feature engineering polynomial interaction")
        
        # 모든 검색 쿼리를 결합
        combined_query = " ".join(search_queries)
        if not combined_query:
            combined_query = "data preprocessing cleaning transformation"
        
        # 관련 코드 검색
        relevant_codes = self.search_relevant_codes(combined_query, top_k=8)
        
        # 검색된 코드들을 문자열로 정리
        code_examples = []
        for item in relevant_codes:
            code_examples.append(f"""
## {item['technique']['name']}
**Description**: {item['technique']['description']}
**Use Case**: {item['technique']['use_case']}
**Code**:
```python
{item['technique']['code']}
```
""")
        
        # LLM 프롬프트 생성
        prompt = f"""
You are a data preprocessing expert. Based on the EDA analysis results and planning information, generate Python preprocessing code using the provided code examples as reference.

=== EDA Analysis Results ===
{eda_results.get('text_analysis', '')}

=== Missing Value Analysis ===
{eda_results.get('null_analysis_text', '')}

=== Outlier Analysis ===
{eda_results.get('outlier_analysis_text', '')}

=== Categorical Analysis ===
{eda_results.get('cate_analysis_text', '')}

=== Numeric Analysis ===
{eda_results.get('numeric_analysis_text', '')}

=== Correlation Analysis ===
{eda_results.get('corr_analysis_text', '')}

=== Planning Information ===
Tasks to perform: {planning_info.get('tasks', [])}
Priority order: {planning_info.get('priority', [])}
Rationale: {planning_info.get('rationale', '')}

=== Relevant Code Examples ===
{''.join(code_examples)}

=== Requirements ===
1. Generate complete, executable Python code for data preprocessing
2. Use the provided code examples as reference but adapt them to the specific data characteristics
3. Follow the task priority order from planning information
4. Include appropriate comments explaining each step
5. Handle edge cases and add error checking where necessary
6. Ensure the code is optimized for the specific data patterns identified in EDA

Please generate the preprocessing code:

```python
# Data Preprocessing Pipeline
# Generated based on EDA analysis and KB examples

import pandas as pd
import numpy as np

# Your preprocessing code here...
```
"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            generated_code = response.content
            
            # 코드 블록에서 실제 코드만 추출
            if "```python" in generated_code:
                generated_code = generated_code.split("```python")[1].split("```")[0].strip()
            elif "```" in generated_code:
                generated_code = generated_code.split("```")[1].split("```")[0].strip()
            
            return generated_code
        
        except Exception as e:
            print(f"코드 생성 오류: {e}")
            return "# 코드 생성에 실패했습니다."


def kb_rag_agent_function(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    KB RAG Agent 함수
    EDA 결과와 계획 정보를 받아 RAG 기반 전처리 코드를 생성합니다.
    """
    try:
        # KB RAG Agent 인스턴스 생성
        kb_agent = KBRAGAgent()
        
        # EDA 결과 추출
        eda_results = {
            'text_analysis': inputs.get('text_analysis', ''),
            'null_analysis_text': inputs.get('null_analysis_text', ''),
            'outlier_analysis_text': inputs.get('outlier_analysis_text', ''),
            'cate_analysis_text': inputs.get('cate_analysis_text', ''),
            'numeric_analysis_text': inputs.get('numeric_analysis_text', ''),
            'corr_analysis_text': inputs.get('corr_analysis_text', ''),
        }
        
        # 계획 정보 추출
        planning_info = inputs.get('planning_info', {
            'tasks': ['missing_values', 'outliers', 'categorical_encoding', 'scaling'],
            'priority': ['missing_values', 'outliers', 'categorical_encoding', 'scaling'],
            'rationale': 'Standard preprocessing pipeline based on EDA results'
        })
        
        # RAG 기반 코드 생성
        generated_code = kb_agent.generate_code_with_rag(eda_results, planning_info)
        
        print("KB RAG Agent: 전처리 코드 생성 완료")
        
        return {
            **inputs,
            'generated_preprocessing_code': generated_code,
            'rag_relevant_codes': kb_agent.search_relevant_codes(
                " ".join(planning_info.get('tasks', [])), top_k=5
            )
        }
        
    except Exception as e:
        print(f"KB RAG Agent 오류: {e}")
        return {
            **inputs,
            'generated_preprocessing_code': "# KB RAG Agent 실행 중 오류가 발생했습니다.",
            'error': str(e)
        }


# LangGraph 노드로 사용할 수 있는 함수
kb_rag_agent = RunnableLambda(kb_rag_agent_function)