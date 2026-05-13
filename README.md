# Bomber Escape RL - Gradio Web Version

기존 Pygame 창 실행 방식의 강화학습 게임을 Gradio 웹 프로그램으로 바꾼 버전입니다.

## 로컬 실행

```bash
pip install -r requirements.txt
python app.py
```

실행 후 브라우저에서 표시되는 주소로 접속하면 됩니다.

## Hugging Face Spaces 배포

1. 새 Space 생성
2. SDK를 `Gradio`로 선택
3. 이 폴더의 `app.py`, `requirements.txt`를 업로드
4. 자동 빌드가 끝나면 웹에서 실행됩니다.

## 조작법

- `선택한 만큼 학습`: 지정한 episode 수만큼 Q-learning 추가 학습
- `데모 한 스텝`: 현재 학습된 Q-table로 한 번 이동
- `데모 한 판 끝까지`: 탈출/폭발/시간초과까지 자동 진행
- `새 데모`: 새 시작 위치와 출구 위치로 데모 초기화
- `전체 리셋`: 에이전트와 통계를 초기화한 뒤 기본 사전 학습 다시 수행
