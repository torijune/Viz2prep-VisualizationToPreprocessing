"""
Viz2prep: Visualization to Preprocessing
멀티모달 LLM 에이전트를 사용한 데이터 전처리 성능 평가 시스템

메인 실행 파일
"""

import os
import sys
from dotenv import load_dotenv
from graph_builder import run_workflow

# 환경 변수 로드
load_dotenv()

def download_titanic_dataset():
    """
    Titanic 데이터셋을 다운로드합니다.
    """
    titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    
    if not os.path.exists("titanic.csv"):
        print("Titanic 데이터셋을 다운로드 중...")
        try:
            import requests
            response = requests.get(titanic_url, verify=False)
            response.raise_for_status()
            with open("titanic.csv", "wb") as f:
                f.write(response.content)
            print("다운로드 완료!")
        except Exception as e:
            print(f"다운로드 실패: {e}")
            return False
    else:
        print("Titanic 데이터셋이 이미 존재합니다.")
    
    return True


def check_environment():
    """
    환경 설정을 확인합니다.
    """
    print("환경 설정 확인 중...")
    
    # OpenAI API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("경고: OPENAI_API_KEY가 설정되지 않았습니다.")
        print("멀티모달 LLM 기능이 제한될 수 있습니다.")
        return False
    
    print("환경 설정 확인 완료!")
    return True


def main():
    """
    메인 실행 함수
    """
    print("="*80)
    print("Viz2prep: Visualization to Preprocessing")
    print("멀티모달 LLM 에이전트를 사용한 데이터 전처리 성능 평가")
    print("="*80)
    
    # 환경 설정 확인
    if not check_environment():
        print("환경 설정에 문제가 있습니다. .env 파일을 확인해주세요.")
        return
    
    # Titanic 데이터셋 다운로드
    if not download_titanic_dataset():
        print("데이터셋 다운로드에 실패했습니다.")
        return
    
    try:
        # 워크플로우 실행
        result = run_workflow("titanic.csv")
        
        # 결과 요약 출력
        print("\n" + "="*80)
        print("실행 결과 요약")
        print("="*80)
        
        print(f"📊 원본 데이터 크기: {result['raw_dataframe'].shape}")
        print(f"🔧 전처리된 데이터 크기: {result['preprocessed_dataframe'].shape}")
        
        if 'plot_paths' in result:
            print(f"📈 생성된 시각화 플롯 수: {len(result['plot_paths'])}")
            for i, plot_path in enumerate(result['plot_paths'], 1):
                print(f"   {i}. {plot_path}")
        
        if 'preprocessing_code' in result:
            code_length = len(result['preprocessing_code'])
            print(f"💻 생성된 전처리 코드 길이: {code_length} 문자")
        
        if 'evaluation_results' in result:
            print(f"📋 성능 평가 완료")
            print(f"   - 평가된 모델 수: {len(result['evaluation_results']['raw_performance'])}")
        
        if 'summary_table' in result:
            print("\n📊 성능 비교 결과:")
            print(result['summary_table'])
        
        print("\n✅ 워크플로우 실행이 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"❌ 워크플로우 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 