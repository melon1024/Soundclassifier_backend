#-*- coding: utf-8 -*-
# (한글을 쓰려면 위의 주석이 반드시 필요하다.)
"""
스크립트 구동 방법
$ export FLASK_APP=urban_sound_classifier.py
$ flask run
... Running on http://127.0.0.1:5000
"""

# 텐서플로우를 켠다. numpy도 켠다.
import tensorflow as tf
import numpy as np
import librosa
from flask import Flask, request,jsonify
import os
import sys
import contextlib

#answer = ["물 흐르는 소리","아이 소리","파열음","빗 소리","끓는 소리","알람","전화벨 소리","청소기","헤어드라이어","고양이 소리"]

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield 
    np.set_printoptions(**original)

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name,mono=True)
   # mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=80).T,axis=0)
    print("audio series's num",X.shape,"samplerate",sample_rate)
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate,hop_length=int(sample_rate*0.01),n_fft=int(sample_rate*0.02),n_mfcc=39).T
    npmfcc = np.array(mfccs)
    print(npmfcc.shape)
    return mfccs

n_dim = 39 # 소리파일에서 추출한 데이터의 차원 수(우리는 80차 벡터를 씀)
n_classes = 10 # 결과값으로 분류할 가짓 수(우리는 총 10가지 음향을 분류함)
n_hid1 = 300 # hidden layer 1의 차원 수
n_hid2 = 200 # hidden layer 2의 차원 수
n_hid3 = 100 # hidden layer 3의 차원 수

training_epochs = 5000 # 학습 횟수
learning_rate = 0.01 # 학습 비율
sd = 1 / np.sqrt(n_dim) # standard deviation 표준편차(표본표준편차라 1/root(n))
win_size = 50 # silence part에 잘 적응하도록 하기 위한 window size(50 개/1 sec)

# 입력 데이터 파라미터와 정답 데이터 파라미터를 생성한다.
X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, n_classes])

# 1차 히든 레이어(원소까지 랜덤인 배열을 생성)
W_1 = tf.Variable(tf.random_normal([n_dim, n_hid1], mean=0, stddev=sd), name="w1")
b_1 = tf.Variable(tf.random_normal([n_hid1], mean=0, stddev=sd), name="b1")
h_1 = tf.nn.relu(tf.matmul(X, W_1) + b_1) # 1차 히든레이어는 'Relu' 함수를 쓴다.
#h_1 = tf.nn.sigmoid(tf.matmul(X, W_1) + b_1) # 1차 히든레이어는 '시그모이드' 함수를 쓴다.

# 2차 히든 레이어
W_2 = tf.Variable(tf.random_normal([n_hid1, n_hid2], mean=0, stddev=sd), name="w2")
b_2 = tf.Variable(tf.random_normal([n_hid2], mean=0, stddev=sd), name="b2")
h_2 = tf.nn.tanh(tf.matmul(h_1, W_2) + b_2) # 2차 히든레이어는 '하이퍼볼릭탄젠트' 함수를 쓴다.

# 3차 히든 레이어
W_3 = tf.Variable(tf.random_normal([n_hid2, n_hid3], mean=0, stddev=sd), name="w3")
b_3 = tf.Variable(tf.random_normal([n_hid3], mean=0, stddev=sd), name="b3")
h_3 = tf.nn.relu(tf.matmul(h_2, W_3) + b_3) # 3차 히든레이어는 'Relu' 함수를 쓴다.
#h_3 = tf.nn.sigmoid(tf.matmul(h_2, W_3) + b_3) # 3차 히든레이어는 '시그모이드' 함수를 쓴다.

# 최종 evidence 레이어(?? 이거 뭐라고 불러야하지)
W = tf.Variable(tf.random_normal([n_hid3, n_classes], mean=0, stddev=sd), name="w")
b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd), name="b")
y_ = tf.nn.softmax(tf.matmul(h_3, W) + b) # 소프트맥스 처리한 것 하나 이렇게 총 두개를 만들어둔다.(왜??)

# 세션을 켜고, 초기화한다.
sess=tf.Session() # 이러면 끝나고 세션을 자동으로 닫아준다.
sess.run(tf.initialize_all_variables()) 

# 이미 학습된 모델 데이터를 불러온다.
saver = tf.train.Saver() # 파일을 불러온다
saver.restore(sess, "/home/sovis2016/mymodel.ckpt")

# 플래스크 앱 생성
app = Flask(__name__)
# 업로드 폴더 설정
app.config['UPLOAD_FOLDER'] = './upload'
# 최대 업로드 설정a
app.config['DEBUG']=True
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 * 10

@app.route("/")
def hello():
    return "hello!"

@app.route('/uploads', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file :
            filename = file.filename
            audio_file=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(audio_file)
            mfccs = extract_feature(audio_file)
            np.array(mfccs)
            print("mfcc len : " +str(mfccs.size))
            x_data = np.hstack([mfccs])
            print(x_data,x_data.size)
            print(x_data.shape)
            #
            y_hat = sess.run(y_, feed_dict={X: x_data})
            ans_list = np.argmax(y_hat,1)
            ans_p1 = np.zeros(shape=[10], dtype=np.float32)
            ans_p2 = np.zeros(shape=[10], dtype=np.float32)
            ans_1sec = np.zeros(shape=[10], dtype=np.float32)
            ans_fin = np.zeros(shape=[10], dtype=np.float32)
            #
            j=0 # 아래 for문을 위한 index
            for k in ans_list:
                ans_p2[k] += 1 # 정답을 모은다.
                if(j%(win_size/2) == 0):
                    # 25개(0.5초, Sliding size)를 모았으면, 50개(1초 = Window Size)에 대한 정답을 낸다.
                    ans_1sec = ans_p1 + ans_p2
                    ans_1sec /= ans_1sec.sum()
                    # 최종 정답에 50개의 정답 결과를 반영한다.
                    ans_fin[ans_1sec.argmax()] += 1
                    # 한 칸을 슬라이드 시킨다.
                    ans_p1 = ans_p2
                    ans_p2 = np.zeros(shape=[10], dtype=np.float32)
                j += 1
            # 맨 마지막에 나온 값도 처리를 해줘야한다.
            ans_1sec = ans_p1 + ans_p2
            ans_1sec /= ans_1sec.sum()
            ans_fin[ans_1sec.argmax()] += 1
            ans_fin /= ans_fin.sum()
            # 추측한 정답을 출력한다.
            with printoptions(precision=3, suppress=True):
                print(ans_fin)
            index = ans_fin.argmax()
            print(index)       
            #index = 0
            #
            return jsonify(classification=index.tolist(),prob=ans_fin.tolist())
    return '''
    <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
    <p><input type=file name=file>
    <input type=submit value=Upload>
    </form>
    '''

app.run(host='0.0.0.0',port=5000)
