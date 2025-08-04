"""
평가 에이전트
원본 데이터와 전처리된 데이터의 ML 성능을 비교 평가합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')


def evaluate_performance(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    원본 데이터와 전처리된 데이터의 ML 성능을 비교 평가합니다.
    
    Args:
        inputs: 원본 DataFrame과 전처리된 DataFrame이 포함된 입력 딕셔너리
        
    Returns:
        평가 결과가 포함된 딕셔너리
    """
    raw_df = inputs["dataframe"]
    preprocessed_df = inputs["preprocessed_dataframe"]
    
    # 타겟 변수 설정 (Titanic 데이터셋 기준)
    target_column = None
    
    # 일반적인 타겟 컬럼명들 확인
    possible_targets = ['Survived', 'target', 'label', 'class', 'y']
    for col in possible_targets:
        if col in raw_df.columns:
            target_column = col
            break
    
    if target_column is None:
        # 마지막 컬럼을 타겟으로 사용
        target_column = raw_df.columns[-1]
        print(f"타겟 컬럼을 찾을 수 없어 마지막 컬럼 '{target_column}'을 사용합니다.")
    
    print(f"타겟 컬럼: {target_column}")
    
    # 원본 데이터 준비
    X_raw = raw_df.drop(columns=[target_column])
    y_raw = raw_df[target_column]
    
    # 전처리된 데이터 준비
    X_preprocessed = preprocessed_df.drop(columns=[target_column])
    y_preprocessed = preprocessed_df[target_column]

    print("전처리 전 columns : ", X_raw.columns)
    print("전처리 후 columns : ", X_preprocessed.columns)
    
    # 타겟 컬럼을 정수형으로 강제 변환 (LLM이 float로 변환했을 경우 대비)
    y_preprocessed = y_preprocessed.astype(int)
    
    # 범주형 변수 처리 (원본 데이터)
    categorical_cols = X_raw.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        X_raw_encoded = X_raw.copy()
        for col in categorical_cols:
            X_raw_encoded[col] = pd.Categorical(X_raw_encoded[col]).codes
        X_raw = X_raw_encoded
    
    # 결측값 처리 (원본 데이터)
    X_raw = X_raw.fillna(X_raw.median())
    
    # 데이터 분할
    X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    
    X_prep_train, X_prep_test, y_prep_train, y_prep_test = train_test_split(
        X_preprocessed, y_preprocessed, test_size=0.2, random_state=42, stratify=y_preprocessed
    )
    
    # 모델 정의
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    results = {
        'raw_performance': {},
        'preprocessed_performance': {},
        'improvement': {}
    }
    
    print("\n=== 성능 평가 시작 ===")
    
    for model_name, model in models.items():
        print(f"\n--- {model_name} 모델 평가 ---")
        
        # 원본 데이터 성능
        print("원본 데이터로 학습 중...")
        model_raw = model.__class__(**model.get_params())
        model_raw.fit(X_raw_train, y_raw_train)
        y_raw_pred = model_raw.predict(X_raw_test)
        y_raw_pred_proba = model_raw.predict_proba(X_raw_test)[:, 1] if len(np.unique(y_raw)) == 2 else None
        
        raw_accuracy = accuracy_score(y_raw_test, y_raw_pred)
        raw_f1 = f1_score(y_raw_test, y_raw_pred, average='weighted')
        raw_auc = roc_auc_score(y_raw_test, y_raw_pred_proba) if y_raw_pred_proba is not None else None
        
        results['raw_performance'][model_name] = {
            'accuracy': raw_accuracy,
            'f1_score': raw_f1,
            'auc': raw_auc
        }
        
        # 안전한 AUC 출력 함수
        def format_auc(val):
            return f"{val:.4f}" if val is not None else "N/A"

        # 원본 데이터 성능
        print(f"원본 데이터 성능 - Accuracy: {raw_accuracy:.4f}, F1: {raw_f1:.4f}, AUC: {format_auc(raw_auc)}")
        
        # 전처리된 데이터 성능
        print("전처리된 데이터로 학습 중...")
        model_prep = model.__class__(**model.get_params())
        model_prep.fit(X_prep_train, y_prep_train)
        y_prep_pred = model_prep.predict(X_prep_test)
        y_prep_pred_proba = model_prep.predict_proba(X_prep_test)[:, 1] if len(np.unique(y_preprocessed)) == 2 else None
        
        prep_accuracy = accuracy_score(y_prep_test, y_prep_pred)
        prep_f1 = f1_score(y_prep_test, y_prep_pred, average='weighted')
        prep_auc = roc_auc_score(y_prep_test, y_prep_pred_proba) if y_prep_pred_proba is not None else None
        
        results['preprocessed_performance'][model_name] = {
            'accuracy': prep_accuracy,
            'f1_score': prep_f1,
            'auc': prep_auc
        }
        
        print(f"전처리된 데이터 성능 - Accuracy: {prep_accuracy:.4f}, F1: {prep_f1:.4f}, AUC: {format_auc(prep_auc)}")
        
        # 성능 개선 계산
        accuracy_improvement = ((prep_accuracy - raw_accuracy) / raw_accuracy) * 100 if raw_accuracy > 0 else 0
        f1_improvement = ((prep_f1 - raw_f1) / raw_f1) * 100 if raw_f1 > 0 else 0
        auc_improvement = ((prep_auc - raw_auc) / raw_auc) * 100 if raw_auc and prep_auc else 0
        
        results['improvement'][model_name] = {
            'accuracy_improvement': accuracy_improvement,
            'f1_improvement': f1_improvement,
            'auc_improvement': auc_improvement
        }
        
        print(f"성능 개선 - Accuracy: {accuracy_improvement:+.2f}%, F1: {f1_improvement:+.2f}%, AUC: {auc_improvement:+.2f}%")
    
    # 결과 요약 테이블 생성
    summary_table = create_summary_table(results)
    
    print("\n=== 성능 평가 완료 ===")
    print("\n" + "="*80)
    print("성능 비교 요약")
    print("="*80)
    print(summary_table)
    
    return {
        "evaluation_results": results,
        "summary_table": summary_table,
        "raw_dataframe": raw_df,
        "preprocessed_dataframe": preprocessed_df
    }


def create_summary_table(results: Dict[str, Any]) -> str:
    """
    성능 평가 결과를 요약 테이블로 생성합니다.
    
    Args:
        results: 평가 결과 딕셔너리
        
    Returns:
        요약 테이블 문자열
    """
    table_lines = []
    table_lines.append("모델명".ljust(20) + "데이터".ljust(15) + "Accuracy".ljust(12) + "F1-Score".ljust(12) + "AUC".ljust(12))
    table_lines.append("-" * 80)
    
    for model_name in results['raw_performance'].keys():
        # 원본 데이터 결과
        raw_perf = results['raw_performance'][model_name]
        table_lines.append(
            model_name.ljust(20) + 
            "원본".ljust(15) + 
            f"{raw_perf['accuracy']:.4f}".ljust(12) + 
            f"{raw_perf['f1_score']:.4f}".ljust(12) + 
            f"{raw_perf['auc']:.4f}".ljust(12) if raw_perf['auc'] else "N/A".ljust(12)
        )
        
        # 전처리된 데이터 결과
        prep_perf = results['preprocessed_performance'][model_name]
        table_lines.append(
            "".ljust(20) + 
            "전처리".ljust(15) + 
            f"{prep_perf['accuracy']:.4f}".ljust(12) + 
            f"{prep_perf['f1_score']:.4f}".ljust(12) + 
            f"{prep_perf['auc']:.4f}".ljust(12) if prep_perf['auc'] else "N/A".ljust(12)
        )
        
        # 개선률
        improvement = results['improvement'][model_name]
        table_lines.append(
            "".ljust(20) + 
            "개선률".ljust(15) + 
            f"{improvement['accuracy_improvement']:+.2f}%".ljust(12) + 
            f"{improvement['f1_improvement']:+.2f}%".ljust(12) + 
            f"{improvement['auc_improvement']:+.2f}%".ljust(12)
        )
        table_lines.append("")  # 빈 줄 추가
    
    return "\n".join(table_lines)


# LangGraph 노드로 사용할 수 있는 함수
evaluator = RunnableLambda(evaluate_performance) 