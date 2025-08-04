# Viz2prep: Visualization to Preprocessing

멀티모달 LLM 에이전트를 사용하여 시각화 기반 데이터 전처리 성능을 평가하는 프로젝트입니다.

## 프로젝트 목표

통계 요약과 시각화 플롯을 모두 제공받은 멀티모달 LLM이 유용한 전처리 코드를 생성할 수 있는지 평가합니다.

## 아키텍처

1. **DataLoader**: CSV 데이터를 로드
2. **StatisticsAgent**: 데이터 통계 요약 생성
3. **VisualizationAgent**: EDA 플롯 생성
4. **PreprocessingAgent**: 멀티모달 LLM을 사용한 전처리 코드 생성
5. **Evaluator**: ML 성능 비교 평가

## 설치 및 실행

```bash
pip install -r requirements.txt
python main.py
```

## 환경 변수 설정

`.env` 파일을 생성하고 OpenAI API 키를 설정하세요:

```
OPENAI_API_KEY=your_api_key_here
``` 