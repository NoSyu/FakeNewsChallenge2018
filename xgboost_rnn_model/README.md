## XGBoost, RNN ensemble model

- 이 프로그램은 Fake News Dectection 2017의 데이터를 이용해 학습, 존재하는 test 데이터로 검출 성능을 확인하는 프로그램입니다.

- 이 모델은 python3.5 버전으로 작성되었습니다.

- 프로그램에 알맞은 환경설정을 위해 virtualenv나 anaconda와 같은 가상환경에서 실행하는 것을 권장합니다.

<pre>
<code>
# 가상환경에서 requirements.txt 내에 명시된 라이브러리들을 설치

pip install --requirement requirements.txt

# 또는

pip install -r requirements.txt

</code>
</pre>

다음의 feature 파일을 받아 압축해제하고 3개의 파일을 xgboost_rnn_model 폴더로 복사 또는 이동.

https://drive.google.com/file/d/1y51XCVP25SI0yDQeo1RO8hMsVDFkVajP/view?usp=sharing

xgboost_rnn_model 경로에서
<pre>
<code>

python3.5 demo_main.py

</code>
</pre>
을 입력해 실행가능합니다.

- 실행화면
<pre>
<code>


#######################################
##         Fake news detector        ##
#######################################
1. Index를 이용한 파일 단위 detection
2. Test 전체 결과 출력
#######################################

번호를 입력하세요. 나가기(Q or q) : q

</code>
</pre>
