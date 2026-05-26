from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from scipy.stats import skew, kurtosis
import io
import os

app = Flask(__name__)

# 모델 및 스케일러 로드
model_svm = joblib.load('saved_svm_model.pkl')
scaler = joblib.load('saved_scaler.pkl')

# 반려견의 평소 정상 걸음 점수를 기억할 변수
global_baseline_score = None

def extract_features(df, fs=100, win_sec=1.0):
    x_data, y_data, z_data = df['ABack_x'].values, df['ABack_y'].values, df['ABack_z'].values
    win_size = int(fs * win_sec)
    features = []
    
    for s in range(0, len(z_data) - win_size, win_size):
        row = []
        for win in [x_data[s:s+win_size], y_data[s:s+win_size], z_data[s:s+win_size]]:
            row.extend([
                np.std(win), np.max(win)-np.min(win), np.sqrt(np.mean(win**2)), 
                np.percentile(win, 75)-np.percentile(win, 25), np.mean(np.abs(win-np.mean(win))), 
                skew(win), kurtosis(win)
            ])
        features.append(row)
    return pd.DataFrame(features)

@app.route('/')
def index():
    return render_template('index.html')

# 1단계: 평소 걸음 점수 등록 (영점 설정)
@app.route('/calibrate', methods=['POST'])
def calibrate():
    global global_baseline_score
    
    file = request.files.get('file')
    preset = request.form.get('preset') 

    try:
        if preset:
            filename = f"{preset}_정상.csv"
            if not os.path.exists(filename):
                return jsonify({"error": f"서버에 {filename} 파일이 없습니다."}), 404
            df = pd.read_csv(filename)
        elif file and file.filename != '':
            df = pd.read_csv(io.BytesIO(file.read()))
        else:
            return jsonify({"error": "보행 데이터를 선택해주세요."}), 400

        new_df = extract_features(df)
        if new_df.empty:
            return jsonify({"error": "데이터 전처리에 실패했습니다."}), 400
            
        new_scaled = scaler.transform(new_df)
        preds = model_svm.predict(new_scaled)
        
        # 평소 상태의 기준 점수 산출
        global_baseline_score = float(np.mean(preds) * 100)
        
        return jsonify({
            "message": "평소 걸음 점수 등록이 완료되었습니다.",
            "baseline_score": round(global_baseline_score, 2)
        })
    except Exception as e:
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500

# 2단계: 실시간 걸음 변화 분석
@app.route('/analyze', methods=['POST'])
def analyze_csv():
    global global_baseline_score
    
    file = request.files.get('file')
    preset = request.form.get('preset') 

    if global_baseline_score is None:
        return jsonify({"error": "먼저 [1단계]에서 평소 걸음 데이터를 등록해 주세요."}), 400

    try:
        if preset:
            filename = f"{preset}.csv"
            if not os.path.exists(filename):
                return jsonify({"error": f"서버에 {filename} 파일이 없습니다."}), 404
            df = pd.read_csv(filename)
        elif file and file.filename != '':
            df = pd.read_csv(io.BytesIO(file.read()))
        else:
            return jsonify({"error": "분석할 데이터를 선택해주세요."}), 400

        new_df = extract_features(df)
        if new_df.empty:
            return jsonify({"error": "데이터 전처리에 실패했습니다."}), 400
            
        new_scaled = scaler.transform(new_df)
        preds = model_svm.predict(new_scaled)
        
        current_score = float(np.mean(preds) * 100)
        
        # 💡 평소 걸음 대비 순수하게 이상 징후가 얼마나 늘었는지 계산
        relative_score = current_score - global_baseline_score
        if relative_score < 0:
            relative_score = 0.0
            
        # 직관적인 변동량 기준 경고 커트라인
        t_suspect = 20.0  # 평소보다 이상 징후 20% 이상 증가 시 주의
        t_danger = 50.0   # 평소보다 이상 징후 50% 이상 증가 시 위험
        
        status = ""
        message = ""
        if relative_score >= t_danger:
            status = "위험 (경고 기준 초과)"
            message = f"평소 상태와 비교했을 때 이상 보행 패턴이 {round(relative_score, 1)}% 급증했습니다. 슬개골 탈구 진행(2기 이상)이 강력히 의심됩니다."
        elif relative_score >= t_suspect:
            status = "주의 (의심 기준 초과)"
            message = f"평소 상태 대비 미세한 걸음걸이 이탈({round(relative_score, 1)}%)이 감지되었습니다. 초기 파행(1기) 의심 단계입니다."
        else:
            status = "안정 (정상 범위)"
            message = f"현재 반려견의 평소 정상 걸음 리듬과 완벽하게 일치합니다. 매우 건강하고 안정적인 상태입니다. (변화량: {round(relative_score, 1)}%)"
            
        display_name = file.filename if file else f"예시 데이터 ({preset}.csv)"

        return jsonify({
            "filename": display_name,
            "baseline_score": round(global_baseline_score, 2),
            "current_score": round(current_score, 2),
            "relative_score": round(relative_score, 2),
            "status": status,
            "message": message
        })
        
    except Exception as e:
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500

if __name__ == '__main__':
    # 외부 접속 허용 코드는 유지하되 발표 컴퓨터 단독 시연을 권장합니다.
    app.run(host='0.0.0.0', debug=True, port=5000)