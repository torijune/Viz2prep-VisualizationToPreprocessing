#!/usr/bin/env python3
"""
데이터 전처리 워크플로우 메인 실행 파일
사용법: 
  python main.py --dataset datasets/Iris/Iris.csv
  python main.py --X datasets/Iris/X.csv --Y datasets/Iris/Y.csv
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from workflow_graph import build_specialized_workflow
from workflow_state import WorkflowState

def load_dataset(dataset_path: str) -> tuple:
    """
    데이터셋을 로드하고 X, Y로 분리하는 함수
    
    Args:
        dataset_path (str): 데이터셋 파일 경로
        
    Returns:
        tuple: (X_df, Y_df) 특성 변수와 타겟 변수 데이터프레임
        
    Raises:
        FileNotFoundError: 파일을 찾을 수 없는 경우
        Exception: 기타 로드 오류
    """
    try:
        # 파일 존재 여부 확인
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {dataset_path}")
        
        # 파일 확장자 확인
        file_extension = Path(dataset_path).suffix.lower()
        
        # 확장자에 따른 로드 방법 결정
        if file_extension == '.csv':
            df = pd.read_csv(dataset_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(dataset_path)
        elif file_extension == '.json':
            df = pd.read_json(dataset_path)
        elif file_extension == '.parquet':
            df = pd.read_parquet(dataset_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {file_extension}")
        
        print(f"✅ [LOAD] 데이터셋 로드 완료: {df.shape}")
        print(f"📊 [LOAD] 컬럼: {list(df.columns)}")
        print(f"📊 [LOAD] 데이터 타입: {df.dtypes.to_dict()}")
        print(f"📊 [LOAD] 결측값: {df.isnull().sum().sum()}개")
        
        # X, Y 분리
        X, Y = separate_features_and_target(df, dataset_path)
        
        return X, Y
        
    except FileNotFoundError as e:
        print(f"❌ [ERROR] {e}")
        return None, None
    except Exception as e:
        print(f"❌ [ERROR] 데이터 로드 오류: {e}")
        return None, None

def load_separate_datasets(X_path: str, Y_path: str) -> tuple:
    """
    X와 Y를 별도의 파일에서 로드하는 함수
    
    Args:
        X_path (str): X 데이터셋 파일 경로
        Y_path (str): Y 데이터셋 파일 경로
        
    Returns:
        tuple: (X_df, Y_df) 특성 변수와 타겟 변수 데이터프레임
    """
    try:
        # X 데이터 로드
        if not os.path.exists(X_path):
            raise FileNotFoundError(f"X 데이터셋 파일을 찾을 수 없습니다: {X_path}")
        
        X_extension = Path(X_path).suffix.lower()
        if X_extension == '.csv':
            X = pd.read_csv(X_path)
        elif X_extension in ['.xlsx', '.xls']:
            X = pd.read_excel(X_path)
        elif X_extension == '.json':
            X = pd.read_json(X_path)
        elif X_extension == '.parquet':
            X = pd.read_parquet(X_path)
        else:
            raise ValueError(f"지원하지 않는 X 파일 형식입니다: {X_extension}")
        
        # Y 데이터 로드
        if not os.path.exists(Y_path):
            raise FileNotFoundError(f"Y 데이터셋 파일을 찾을 수 없습니다: {Y_path}")
        
        Y_extension = Path(Y_path).suffix.lower()
        if Y_extension == '.csv':
            Y = pd.read_csv(Y_path)
        elif Y_extension in ['.xlsx', '.xls']:
            Y = pd.read_excel(Y_path)
        elif Y_extension == '.json':
            Y = pd.read_json(Y_path)
        elif Y_extension == '.parquet':
            Y = pd.read_parquet(Y_path)
        else:
            raise ValueError(f"지원하지 않는 Y 파일 형식입니다: {Y_extension}")
        
        print(f"✅ [LOAD] X 데이터셋 로드 완료: {X.shape}")
        print(f"📊 [LOAD] X 컬럼: {list(X.columns)}")
        print(f"📊 [LOAD] X 데이터 타입: {X.dtypes.to_dict()}")
        print(f"📊 [LOAD] X 결측값: {X.isnull().sum().sum()}개")
        
        print(f"✅ [LOAD] Y 데이터셋 로드 완료: {Y.shape}")
        print(f"📊 [LOAD] Y 컬럼: {list(Y.columns)}")
        print(f"📊 [LOAD] Y 데이터 타입: {Y.dtypes.to_dict()}")
        print(f"📊 [LOAD] Y 결측값: {Y.isnull().sum().sum()}개")
        
        return X, Y
        
    except FileNotFoundError as e:
        print(f"❌ [ERROR] {e}")
        return None, None
    except Exception as e:
        print(f"❌ [ERROR] 데이터 로드 오류: {e}")
        return None, None

def separate_features_and_target(df: pd.DataFrame, dataset_path: str = None) -> tuple:
    """
    데이터프레임을 특성 변수(X)와 타겟 변수(Y)로 분리합니다.
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        dataset_path (str): 데이터셋 파일 경로 (선택사항)
        
    Returns:
        tuple: (X_df, Y_df) 특성 변수와 타겟 변수 데이터프레임
    """
    # 일반적인 타겟 변수 컬럼명들
    target_columns = [
        'target', 'target_variable', 'label', 'class', 'y', 'Y',
        'income', 'salary', 'price', 'value', 'result', 'outcome',
        'survived', 'survival', 'death', 'alive',
        'default', 'churn', 'fraud', 'spam',
        'diagnosis', 'disease', 'cancer', 'malignant',
        'species', 'type', 'category'
    ]
    
    # 데이터셋별 특수한 타겟 변수 처리
    dataset_specific_targets = {
        'adult': ['income'],
        'iris': ['species'],
        'titanic': ['survived'],
        'breast_cancer': ['diagnosis'],
        'diabetes': ['outcome'],
        'heart': ['target'],
        'wine': ['target'],
        'digits': ['target']
    }
    
    # 파일명에서 데이터셋 타입 추측
    dataset_type = None
    if dataset_path:
        file_name = Path(dataset_path).stem.lower()
        for key in dataset_specific_targets.keys():
            if key in file_name:
                dataset_type = key
                break
    
    # 타겟 변수 찾기
    target_col = None
    
    # 1. 데이터셋별 특수한 타겟 변수 확인
    if dataset_type and dataset_type in dataset_specific_targets:
        for col in dataset_specific_targets[dataset_type]:
            if col in df.columns:
                target_col = col
                break
    
    # 2. 일반적인 타겟 변수명 확인
    if target_col is None:
        for col in target_columns:
            if col in df.columns:
                target_col = col
                break
    
    # 3. 마지막 컬럼을 타겟 변수로 사용 (기본값)
    if target_col is None:
        target_col = df.columns[-1]
        print(f"⚠️  [WARNING] 명시적인 타겟 변수를 찾을 수 없어 마지막 컬럼 '{target_col}'을 타겟 변수로 사용합니다.")
    
    # X, Y 분리
    Y = df[[target_col]].copy()
    X = df.drop(columns=[target_col]).copy()
    
    print(f"📊 [SEPARATE] X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"📊 [SEPARATE] 특성 변수: {list(X.columns)}")
    print(f"📊 [SEPARATE] 타겟 변수: {list(Y.columns)}")
    
    return X, Y

def run_workflow(dataset_path: str = None, X_path: str = None, Y_path: str = None, query: str = None) -> dict:
    """
    워크플로우를 실행하는 함수
    
    Args:
        dataset_path (str): 통합 데이터셋 파일 경로 (선택사항)
        X_path (str): X 데이터셋 파일 경로 (선택사항)
        Y_path (str): Y 데이터셋 파일 경로 (선택사항)
        query (str): 사용자 쿼리 (기본값: 자동 생성)
        
    Returns:
        dict: 워크플로우 실행 결과
    """
    print("=" * 80)
    print("🚀 데이터 전처리 워크플로우 시작")
    print("=" * 80)
    
    # 1. 데이터셋 로드
    if dataset_path:
        # 통합 데이터셋에서 X, Y 분리
        X, Y = load_dataset(dataset_path)
        if X is None or Y is None:
            return {"error": "데이터셋 로드 실패"}
    elif X_path and Y_path:
        # 별도의 X, Y 파일에서 로드
        X, Y = load_separate_datasets(X_path, Y_path)
        if X is None or Y is None:
            return {"error": "X 또는 Y 데이터셋 로드 실패"}
    else:
        return {"error": "데이터셋 경로가 제공되지 않았습니다. --dataset 또는 --X와 --Y를 사용하세요."}
    
    # 2. 기본 쿼리 생성 (사용자가 제공하지 않은 경우)
    if query is None:
        # 데이터셋 특성에 따른 자동 쿼리 생성
        X_numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        query_parts = []
        if X_numeric_cols:
            query_parts.append(f"수치형 변수({', '.join(X_numeric_cols)}) 정규화")
        if X_categorical_cols:
            query_parts.append(f"범주형 변수({', '.join(X_categorical_cols)}) 원핫 인코딩")
        
        query = f"이 데이터를 머신러닝 모델 학습에 적합하도록 전처리해주세요. {'하고 '.join(query_parts)}를 적용해주세요."
    
    print(f"📝 [QUERY] 사용자 요청: {query}")
    
    # 3. 워크플로우 그래프 구축
    try:
        workflow = build_specialized_workflow()
        print("✅ [GRAPH] 워크플로우 그래프 구축 완료")
    except Exception as e:
        print(f"❌ [ERROR] 워크플로우 그래프 구축 실패: {e}")
        return {"error": f"워크플로우 그래프 구축 실패: {e}"}
    
    # 4. 초기 상태 설정 (X, Y 분리)
    initial_state = {
        "query": query,
        "X": X,
        "Y": Y
    }
    
    # 5. 워크플로우 실행
    print("\n🔄 워크플로우 실행 시작...")
    print("=" * 50)
    
    try:
        result = workflow.invoke(initial_state)
        
        print("\n" + "=" * 80)
        print("🎉 워크플로우 실행 완료!")
        print("=" * 80)
        
        # 6. 결과 출력
        print("\n📝 최종 응답:")
        print(result.get("final_response", "응답을 생성할 수 없습니다."))
        
        # 전처리 요약 출력
        summary = result.get("preprocessing_summary", {})
        if summary:
            print("\n📊 전처리 요약:")
            processing_steps = summary.get('processing_steps', {})
            if processing_steps:
                print(f"   - 성공률: {processing_steps.get('success_rate', 0):.1f}%")
                print(f"   - 성공한 단계: {processing_steps.get('successful_steps', [])}")
                print(f"   - 실패한 단계: {processing_steps.get('failed_steps', [])}")
            
            data_quality = summary.get('data_quality_improvements', {})
            if data_quality:
                print(f"   - 데이터 완전성: {data_quality.get('data_completeness', 0):.1f}%")
                print(f"   - 제거된 결측값: {data_quality.get('missing_values_removed', 0)}개")
        
        # 처리된 데이터프레임 정보
        X_processed = result.get("X_processed")
        Y_processed = result.get("Y_processed")
        
        if X_processed is not None and Y_processed is not None:
            print(f"\n📈 데이터프레임 변화:")
            print(f"   - X: {X.shape} → {X_processed.shape}")
            print(f"   - Y: {Y.shape} → {Y_processed.shape}")
            print(f"   - X 결측값: {X.isnull().sum().sum()} → {X_processed.isnull().sum().sum()}")
            print(f"   - Y 결측값: {Y.isnull().sum().sum()} → {Y_processed.isnull().sum().sum()}")
            
            print(f"\n📋 처리된 X 데이터 샘플:")
            print(X_processed.head(3))
            
            print(f"\n📋 처리된 Y 데이터 샘플:")
            print(Y_processed.head(3))
            
            # 데이터 타입 정보
            print(f"\n📊 최종 X 데이터 타입:")
            for col, dtype in X_processed.dtypes.items():
                print(f"   - {col}: {dtype}")
            
            print(f"\n📊 최종 Y 데이터 타입:")
            for col, dtype in Y_processed.dtypes.items():
                print(f"   - {col}: {dtype}")
        
        return result
        
    except Exception as e:
        print(f"❌ [ERROR] 워크플로우 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"워크플로우 실행 실패: {e}"}

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="데이터 전처리 워크플로우 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 통합 데이터셋 사용
  python main.py --dataset datasets/Iris/Iris.csv
  
  # X, Y 분리 데이터셋 사용
  python main.py --X datasets/Iris/X.csv --Y datasets/Iris/Y.csv
  
  # 쿼리와 함께 사용
  python main.py --dataset datasets/Adult/adult.csv --query "소득 예측을 위한 전처리"
  python main.py --X datasets/Adult/X.csv --Y datasets/Adult/Y.csv --query "소득 예측을 위한 전처리"
        """
    )
    
    # 데이터셋 입력 옵션 (상호 배타적)
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--dataset", 
        type=str,
        help="통합 데이터셋 파일 경로 (CSV, Excel, JSON, Parquet 지원)"
    )
    
    dataset_group.add_argument(
        "--X", 
        type=str,
        help="X 데이터셋 파일 경로 (특성 변수)"
    )
    
    dataset_group.add_argument(
        "--Y", 
        type=str,
        help="Y 데이터셋 파일 경로 (타겟 변수)"
    )
    
    parser.add_argument(
        "--query", 
        type=str, 
        default=None,
        help="사용자 쿼리 (기본값: 자동 생성)"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="결과 저장 경로 (선택사항)"
    )
    
    args = parser.parse_args()
    
    # 입력 검증
    if args.X and not args.Y:
        parser.error("--X가 제공되면 --Y도 필요합니다.")
    if args.Y and not args.X:
        parser.error("--Y가 제공되면 --X도 필요합니다.")
    
    # 전역 변수로 dataset_path 설정 (separate_features_and_target에서 사용)
    global dataset_path
    dataset_path = args.dataset
    
    # 워크플로우 실행
    result = run_workflow(
        dataset_path=args.dataset,
        X_path=args.X,
        Y_path=args.Y,
        query=args.query
    )
    
    # 결과 저장 (요청된 경우)
    if args.output and "error" not in result:
        try:
            X_processed = result.get("X_processed")
            Y_processed = result.get("Y_processed")
            
            if X_processed is not None and Y_processed is not None:
                # X와 Y를 다시 합쳐서 저장
                processed_df = pd.concat([X_processed, Y_processed], axis=1)
                
                output_path = args.output
                file_extension = Path(output_path).suffix.lower()
                
                if file_extension == '.csv':
                    processed_df.to_csv(output_path, index=False)
                elif file_extension in ['.xlsx', '.xls']:
                    processed_df.to_excel(output_path, index=False)
                elif file_extension == '.json':
                    processed_df.to_json(output_path, orient='records')
                elif file_extension == '.parquet':
                    processed_df.to_parquet(output_path, index=False)
                else:
                    print(f"⚠️  [WARNING] 지원하지 않는 출력 형식: {file_extension}")
                    processed_df.to_csv(output_path + '.csv', index=False)
                
                print(f"✅ [SAVE] 결과 저장 완료: {output_path}")
        except Exception as e:
            print(f"❌ [ERROR] 결과 저장 실패: {e}")

if __name__ == "__main__":
    main() 