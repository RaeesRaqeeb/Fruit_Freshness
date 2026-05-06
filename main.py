from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from inference import predict

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def home():
        return """
        <!doctype html>
        <html lang="en">
            <head>
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <title>Fruit Scanner (Keras Edition)</title>
                <style>
                    :root {
                        color-scheme: light;
                        --bg: #f7f7fb;
                        --card: #ffffff;
                        --text: #1f2937;
                        --muted: #6b7280;
                        --accent: #2563eb;
                        --accent-soft: #dbeafe;
                        --border: #e5e7eb;
                        --success: #16a34a;
                        --warn: #dc2626;
                    }
                    * { box-sizing: border-box; }
                    body {
                        margin: 0;
                        font-family: Arial, Helvetica, sans-serif;
                        background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 45%, #fff7ed 100%);
                        color: var(--text);
                        min-height: 100vh;
                        display: grid;
                        place-items: center;
                        padding: 24px;
                    }
                    .app {
                        width: 100%;
                        max-width: 720px;
                        background: rgba(255, 255, 255, 0.85);
                        backdrop-filter: blur(10px);
                        border: 1px solid rgba(229, 231, 235, 0.9);
                        border-radius: 24px;
                        box-shadow: 0 20px 60px rgba(15, 23, 42, 0.12);
                        padding: 28px;
                    }
                    h1 {
                        margin: 0 0 8px;
                        font-size: 2rem;
                    }
                    p { margin: 0; color: var(--muted); }
                    .panel {
                        margin-top: 24px;
                        display: grid;
                        gap: 16px;
                    }
                    .upload {
                        border: 2px dashed var(--border);
                        border-radius: 18px;
                        padding: 18px;
                        background: #fff;
                    }
                    input[type="file"] {
                        width: 100%;
                        padding: 12px;
                        border: 1px solid var(--border);
                        border-radius: 12px;
                        background: #fafafa;
                    }
                    button {
                        appearance: none;
                        border: 0;
                        background: var(--accent);
                        color: white;
                        padding: 12px 18px;
                        border-radius: 12px;
                        font-weight: 700;
                        cursor: pointer;
                    }
                    button:disabled { opacity: 0.65; cursor: not-allowed; }
                    .result {
                        display: none;
                        border: 1px solid var(--border);
                        border-radius: 18px;
                        background: var(--card);
                        padding: 18px;
                    }
                    .grid {
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 12px;
                        margin-top: 12px;
                    }
                    .stat {
                        padding: 14px;
                        border-radius: 14px;
                        background: #f9fafb;
                        border: 1px solid var(--border);
                    }
                    .label { font-size: 0.82rem; color: var(--muted); margin-bottom: 6px; }
                    .value { font-size: 1.1rem; font-weight: 700; }
                    .fresh { color: var(--success); }
                    .rotten { color: var(--warn); }
                    .unknown { color: var(--muted); }
                    .status { margin-top: 12px; font-size: 0.95rem; color: var(--muted); }
                    .preview {
                        width: 100%;
                        max-height: 280px;
                        object-fit: contain;
                        display: none;
                        border-radius: 16px;
                        border: 1px solid var(--border);
                        background: white;
                    }
                    .error { color: var(--warn); margin-top: 12px; display: none; }
                    @media (max-width: 640px) {
                        .app { padding: 20px; }
                        .grid { grid-template-columns: 1fr; }
                    }
                </style>
            </head>
            <body>
                <main class="app">
                    <h1>Fruit Scanner</h1>
                    <p>Powered by Keras CNN Classification</p>

                    <section class="panel">
                        <div class="upload">
                            <input id="fileInput" type="file" accept="image/*" />
                            <div style="display:flex; gap:12px; margin-top:12px; flex-wrap:wrap; align-items:center;">
                                <button id="scanBtn">Scan Fruit</button>
                                <span id="loadingText" class="status" style="display:none;">Analyzing with Keras...</span>
                            </div>
                        </div>

                        <img id="preview" class="preview" alt="Preview" />

                        <div id="errorBox" class="error"></div>

                        <div id="resultBox" class="result">
                            <h2 style="margin:0;">Keras Prediction</h2>
                            <div class="grid">
                                <div class="stat">
                                    <div class="label">Detected Fruit</div>
                                    <div id="fruitValue" class="value">-</div>
                                </div>
                                <div class="stat">
                                    <div class="label">Freshness Status</div>
                                    <div id="statusValue" class="value">-</div>
                                </div>
                            </div>

                            <div id="modelsDetail" style="margin-top:12px;">
                                <div class="grid">
                                    <div class="stat" id="kerasBlock">
                                        <div class="label">Confidence Score</div>
                                        <div id="kConf" class="value">-</div>
                                    </div>
                                    <div class="stat">
                                        <div class="label">Model Source</div>
                                        <div class="value">TensorFlow/Keras</div>
                                    </div>
                                </div>

                                <div id="kerasTop3" style="margin-top:10px;"></div>
                            </div>

                            <div id="statusHint" class="status"></div>
                        </div>
                    </section>
                </main>

                <script>
                    const fileInput = document.getElementById('fileInput');
                    const scanBtn = document.getElementById('scanBtn');
                    const loadingText = document.getElementById('loadingText');
                    const preview = document.getElementById('preview');
                    const errorBox = document.getElementById('errorBox');
                    const resultBox = document.getElementById('resultBox');
                    const fruitValue = document.getElementById('fruitValue');
                    const statusValue = document.getElementById('statusValue');
                    const statusHint = document.getElementById('statusHint');

                    fileInput.addEventListener('change', () => {
                        const file = fileInput.files && fileInput.files[0];
                        if (!file) {
                            preview.style.display = 'none';
                            return;
                        }
                        const reader = new FileReader();
                        reader.onload = () => {
                            preview.src = reader.result;
                            preview.style.display = 'block';
                        };
                        reader.readAsDataURL(file);
                    });

                    scanBtn.addEventListener('click', async () => {
                        const file = fileInput.files && fileInput.files[0];
                        errorBox.style.display = 'none';
                        resultBox.style.display = 'none';

                        if (!file) {
                            errorBox.textContent = 'Please choose an image first.';
                            errorBox.style.display = 'block';
                            return;
                        }

                        const formData = new FormData();
                        formData.append('file', file);

                        scanBtn.disabled = true;
                        loadingText.style.display = 'inline';

                        try {
                            const response = await fetch('/api/scan-fruit', {
                                method: 'POST',
                                body: formData,
                            });

                            if (!response.ok) {
                                const text = await response.text();
                                throw new Error(text || 'Request failed');
                            }

                            const data = await response.json();
                            const normalizeStatus = (status) => (status || '').toLowerCase();

                            if (data.keras) {
                                // Update top-level values using Keras results[cite: 3]
                                fruitValue.textContent = data.keras.fruit_type || '-';
                                statusValue.textContent = data.keras.status || '-';
                                
                                const statusClass = normalizeStatus(data.keras.status) === 'fresh'
                                    ? 'fresh'
                                    : normalizeStatus(data.keras.status) === 'rotten'
                                        ? 'rotten'
                                        : 'unknown';
                                statusValue.className = 'value ' + statusClass;

                                // Update confidence
                                document.getElementById('kConf').textContent = (data.keras.confidence * 100).toFixed(1) + '%';

                                // Render top-3 alternatives[cite: 3]
                                const top3 = data.keras.top3 || [];
                                const container = document.getElementById('kerasTop3');
                                container.innerHTML = '<div class="label">Probability Breakdown (Top-3)</div>' + top3.map(item => {
                                    return `<div style="padding:8px; border-radius:8px; background:#fff; border:1px solid #eee; margin-top:6px;">
                                                <div style="font-weight:700">${item.fruit_type} <span style="font-weight:400; color:#6b7280">(${item.status})</span></div>
                                                <div style="font-size:0.9rem; color:#6b7280">Confidence: ${ (item.probability*100).toFixed(1) }%</div>
                                            </div>`;
                                }).join('');

                                statusHint.textContent = data.keras.rejected 
                                    ? 'Result flagged: Confidence score is below the rejection threshold.' 
                                    : 'Analysis complete using Keras CNN.';
                                
                                resultBox.style.display = 'block';
                            } else {
                                throw new Error('Keras model output not found in response.');
                            }

                        } catch (error) {
                            errorBox.textContent = error.message || 'Something went wrong.';
                            errorBox.style.display = 'block';
                        } finally {
                            scanBtn.disabled = false;
                            loadingText.style.display = 'none';
                        }
                    });
                </script>
            </body>
        </html>
        """

@app.post("/api/scan-fruit")
async def scan_fruit(file: UploadFile = File(...)):
    image_data = await file.read()
    results = predict(image_data)
    print("Inference results:", results)
    return results