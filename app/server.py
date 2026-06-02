# -*- coding: utf-8 -*-
"""Custom Python HTTP server for APEX v2.

Serves static frontend files and exposes a POST /api/predict endpoint
to make real-time student burnout predictions using the scikit-learn pipeline
and Gradient Boosting model package.
"""

import os
import sys
import json
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, HTTPServer
import joblib
import pandas as pd
import numpy as np

# Setup paths relative to this server file
APP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = APP_DIR.parent.resolve()

# Inject project root to sys.path so we can import preprocessing cleanly
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load preprocessor and best model package
PREP_PATH = PROJECT_ROOT / 'models' / 'artifacts' / 'full_preprocessor.pkl'
PACKAGE_PATH = PROJECT_ROOT / 'models' / 'best_model_package.joblib'

try:
    preprocessor = joblib.load(PREP_PATH)
    model_package = joblib.load(PACKAGE_PATH)
    model = model_package['model']
    tuned_threshold = model_package['threshold_decision']['tuned']['threshold']
    print(f"[*] Success: Loaded Gradient Boosting model package from {PACKAGE_PATH.name}")
    print(f"[*] Success: Loaded preprocessor pipeline from {PREP_PATH.name}")
except Exception as e:
    print(f"[!] Error: Could not load model artifacts: {e}")
    preprocessor, model_package, model, tuned_threshold = None, None, None, 0.53

class APEXHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        # Translate path relative to APP_DIR instead of current working directory
        standard_path = super().translate_path(path)
        rel_path = os.path.relpath(standard_path, os.getcwd())
        return os.path.join(str(APP_DIR), rel_path)

    def do_OPTIONS(self):
        # Support CORS preflight requests
        self.send_response(200, "OK")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        error_log_path = APP_DIR / 'server_errors.log'
        try:
            # Log incoming POST request details for debugging
            with open(error_log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n=== Incoming POST request to {self.path} ===\n")
            
            if self.path == '/api/predict':
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                
                # Load JSON request body
                inputs = json.loads(post_data.decode('utf-8'))
                
                # Check if model is loaded properly
                if not model or not preprocessor:
                    raise ValueError("Model artifacts are currently not loaded on this server.")

                # Assemble Lean Core input schema matching preprocessing.py exactly
                input_df = pd.DataFrame({
                    'Okunan Bölüm': [inputs.get('major')],
                    'Sınıf Düzeyi': [inputs.get('year_of_study')],
                    'Haftalık AI Saati': [float(inputs.get('weekly_ai_hours', 10.0))],
                    'Birincil Kullanım Amacı': [inputs.get('use_case')],
                    'Prompt Yazma Becerisi': [np.nan if inputs.get('prompt_skill') == 'Bilinmiyor' else inputs.get('prompt_skill')],
                    'Araç Çeşitliliği': [float(inputs.get('tool_diversity', 3))],
                    'Ücretli Abonelik': [1.0 if inputs.get('paid_sub') == 'Var' else 0.0],
                    'Geleneksel Çalışma Saati': [float(inputs.get('traditional_study_hours', 12.0))],
                    'Algılanan AI Bağımlılığı': [float(inputs.get('ai_dependency', 4.0))],
                    'Kurum Politikası': [inputs.get('policy')],
                    'Sınav Kaygı Düzeyi': [float(inputs.get('exam_anxiety', 5.0))],
                    'Beceri Kalıcılık Skoru': [np.nan if inputs.get('retention_unk', False) else float(inputs.get('retention', 75.0))],
                })
                
                # Apply ColumnTransformer Pipeline preprocessing
                processed_df = preprocessor.transform(input_df)
                
                # Compute probability for burnout
                prob = model.predict_proba(processed_df)
                class_to_col = {cls: i for i, cls in enumerate(model.classes_)}
                prob_high = float(prob[0][class_to_col[1]]) # Probability for Yüksek (High) risk
                
                # Determine risk class based on tuned decision threshold
                is_high_risk = prob_high >= tuned_threshold
                result_label = "Yüksek" if is_high_risk else "Düşük"
                
                # Extract input variables for rich, personalized academic suggestions
                weekly_ai_hours = float(inputs.get('weekly_ai_hours', 10.0))
                ai_dependency = float(inputs.get('ai_dependency', 4.0))
                exam_anxiety = float(inputs.get('exam_anxiety', 5.0))
                traditional_study_hours = float(inputs.get('traditional_study_hours', 12.0))
                
                advisors = []
                if is_high_risk:
                    advisors = [
                        f"<strong>Yapay Zeka Dengesi Kurulmalı:</strong> Öğrencinin haftalık YZ kullanım saati (<strong>{weekly_ai_hours} saat</strong>) ve bağımlılık algısı (<strong>{ai_dependency}/10</strong>) oldukça yüksek bir bilişsel yük yaratmaktadır. AI araçlarını ödev veya yazılımlarda doğrudan kopyalama/ezberleme odaklı kullanmak yerine, kütüphanelerde geleneksel çalışma alışkanlıklarını destekleyici yardımcı bir asistan olarak kullanması önerilmelidir.",
                        f"<strong>Sınav Kaygısı Mentörlüğü:</strong> Öğrencinin sınav dönemi kaygısı (<strong>{exam_anxiety}/10</strong>) yüksek riskli tükenmişlik durumunu tetiklemektedir. Okulun rehberlik birimiyle görüşerek sınav anksiyetesini azaltacak nefes egzersizleri, zaman yönetimi ve odaklanma seansları planlanmalıdır.",
                        "<strong>Dönem Sonu Not Takibi & Destek:</strong> GNO değişimi yakından takip edilmelidir. Eğer dönem başından sonuna doğru belirgin bir düşüş gözlenmişse, gelecek dönem için ders yükü azaltılmalı veya ders seçim süreçleri danışman öğretmen kontrolünde yapılmalıdır."
                    ]
                else:
                    advisors = [
                        f"<strong>Sağlıklı Kullanım Teşviki:</strong> Öğrenci, yapay zekayı geleneksel ders çalışma süreleriyle (<strong>{traditional_study_hours} saat/hafta</strong>) dengeli bir biçimde entegre etmiştir. Bu durum, tükenmişlik riskinin oldukça düşük kalmasını sağlamaktadır. Mevcut süreç teşvik edilmeli, prompt becerilerini artıracak ileri seviye atölyelere katılması sağlanmalıdır.",
                        "<strong>Akran Mentörlüğü Desteği:</strong> Akademik başarısı ve YZ kullanım dengesi yüksek olan bu öğrencimiz, bölümünde yapay zeka bağımlılığı veya sınav stresi sebebiyle akademik tükenmişlik yaşayan (yüksek risk grubundaki) arkadaşlarına rehberlik etmesi amacıyla <strong>Akran Mentörlüğü</strong> programlarına dahil edilebilir."
                    ]
                
                # Structure payload
                response_data = {
                    'status': 'success',
                    'prob_high': prob_high,
                    'is_high_risk': is_high_risk,
                    'result_label': result_label,
                    'tuned_threshold': tuned_threshold,
                    'advisors': advisors
                }
                
                # Send JSON response with appropriate headers
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
                return
            else:
                # Send 404 for other POST paths
                self.send_response(404)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': 'Not Found'}).encode('utf-8'))
                return
        except Exception as e:
            import traceback
            try:
                with open(error_log_path, 'a', encoding='utf-8') as f:
                    f.write("=== Exception during POST ===\n")
                    traceback.print_exc(file=f)
            except Exception as log_err:
                print(f"[!] Logging error: {log_err}")
            
            print("[!] Exception during POST processing:")
            traceback.print_exc()
            try:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': str(e)}).encode('utf-8'))
            except Exception:
                pass
            return

def run_server(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, APEXHandler)
    print(f"[*] APEX v2 API Server running at http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down APEX server.")
        httpd.server_close()

if __name__ == '__main__':
    port = 8080
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
