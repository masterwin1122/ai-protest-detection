# app.py — Campus Protest Detection MVP with Video Showcase
# Features: upload -> detect people + face blur -> annotated MP4 with playback + CSV + chart
# Threshold alerts (≥N people for ≥D seconds) -> SNS email with S3 pre-signed links

import os, uuid, csv, json
from typing import List, Dict, Tuple

from flask import Flask, request, send_from_directory, render_template_string, abort
import boto3
import numpy as np
import cv2
from ultralytics import YOLO

import matplotlib
matplotlib.use("Agg")  # headless for servers
import matplotlib.pyplot as plt
import mimetypes

# --------------------
# Env / AWS clients
# --------------------
AWS_REGION       = os.getenv("AWS_REGION", "us-east-2").strip()
SNS_TOPIC_ARN    = os.getenv("SNS_TOPIC_ARN", "arn:aws:sns:us-east-2:463367047047:aidetect").strip()
S3_BUCKET        = os.getenv("S3_BUCKET", "").strip()
URL_EXPIRES_SECS = int(os.getenv("URL_EXPIRES_SECS", "3600"))

sns = boto3.client("sns", region_name=AWS_REGION) if SNS_TOPIC_ARN else None
s3  = boto3.client("s3",  region_name=AWS_REGION) if S3_BUCKET  else None

def publish_sns(subject: str, message: str) -> Tuple[bool, str]:
    if not sns or not SNS_TOPIC_ARN:
        return False, "SNS not configured - check AWS credentials and topic ARN"
    try:
        print(f"📧 Sending SNS alert to: {SNS_TOPIC_ARN}")
        print(f"📧 Subject: {subject}")
        response = sns.publish(TopicArn=SNS_TOPIC_ARN, Subject=subject, Message=message)
        print(f"📧 SNS Response: {response}")
        return True, f"Message ID: {response.get('MessageId', 'Unknown')}"
    except Exception as e:
        print(f"📧 SNS Error: {str(e)}")
        return False, str(e)

def upload_and_sign(local_path: str, run_id: str) -> str:
    """Upload local file to s3://<bucket>/runs/<run_id>/<filename> and return pre-signed URL."""
    if not s3 or not S3_BUCKET:
        return ""
    key = f"runs/{run_id}/{os.path.basename(local_path)}"
    ctype, _ = mimetypes.guess_type(local_path)
    s3.upload_file(local_path, S3_BUCKET, key, ExtraArgs={"ContentType": ctype or "application/octet-stream"})
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=URL_EXPIRES_SECS,
    )
    return url

# --------------------
# Paths / App config
# --------------------
UPLOAD_DIR = os.path.expanduser("~/uploads")
OUTPUT_DIR = os.path.expanduser("~/outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_MB = 500
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

# YOLO model (CPU-friendly nano)
model = YOLO("yolov8n.pt")

# OpenCV Haar face cascade (fast CPU face blur)
FACE_CASCADE = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)

# --------------------
# HTML templates with video showcase
# --------------------
INDEX_HTML = """
<!doctype html><html><head><title>Campus Protest Detection</title>
<style>
 body{font-family:'Segoe UI',system-ui,sans-serif;margin:40px;max-width:1200px;background:#f8f9fa}
 .header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:30px;border-radius:15px;margin-bottom:25px;text-align:center}
 .box{border:1px solid #e0e0e0;padding:30px;border-radius:15px;margin-bottom:20px;background:white;box-shadow:0 4px 6px rgba(0,0,0,0.1)}
 label{display:inline-block;min-width:200px;font-weight:500;color:#333}
 input[type=number]{width:90px;padding:8px;border:1px solid #ddd;border-radius:6px}
 input[type=file]{padding:10px;border:2px dashed #ddd;border-radius:8px;width:100%;background:#fafafa}
 input[type=checkbox]{transform:scale(1.2);margin-right:8px}
 button{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:15px 30px;border:none;border-radius:8px;font-size:16px;cursor:pointer;transition:transform 0.2s}
 button:hover{transform:translateY(-2px)}
 .config-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px;margin:20px 0}
 .feature-list{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:15px;margin:20px 0}
 .feature{padding:15px;background:#f0f8ff;border-left:4px solid #667eea;border-radius:8px}
 small{color:#666}
</style></head><body>

<div class="header">
  <h1>🎯 Campus Protest Detection System</h1>
  <p>AI-Powered Video Analysis with Real-time Crowd Monitoring</p>
</div>

<div class="box">
  <h3>📤 Upload & Configure</h3>
  <p>Upload a video file (&lt; {{max_mb}} MB) for AI analysis. The system will detect people, blur faces for privacy, and provide comprehensive crowd analytics.</p>
  
  <form action="/process" method="post" enctype="multipart/form-data">
    <div style="margin:20px 0">
      <label>📹 Video File:</label><br>
      <input type="file" name="video" accept="video/*" required>
    </div>

    <div class="config-grid">
      <div>
        <label>🎯 YOLO Confidence (0.1–0.7)</label><br>
        <input type="number" name="conf" step="0.05" min="0.1" max="0.7" value="0.35">
        <small>Higher = fewer false positives</small>
      </div>

      <div>
        <label>👥 People Threshold</label><br>
        <input type="number" name="people_thresh" min="1" max="200" value="15">
        <small>Alert when crowd ≥ N people</small>
      </div>

      <div>
        <label>⏱️ Min Duration (seconds)</label><br>
        <input type="number" name="min_dur" min="1" max="600" value="5">
        <small>Sustained crowd required</small>
      </div>

      <div>
        <label>🔒 Privacy Protection</label><br>
        <input type="checkbox" name="faceblur" checked>
        <small>Automatically blur detected faces</small>
      </div>
    </div>

    <div style="text-align:center;margin-top:30px">
      <button type="submit">🚀 Process Video</button>
    </div>
  </form>
</div>

<div class="box">
  <h3>✨ System Features</h3>
  <div class="feature-list">
    <div class="feature">
      <strong>🤖 AI Detection</strong><br>
      YOLOv8 person detection with configurable confidence
    </div>
    <div class="feature">
      <strong>🔒 Privacy First</strong><br>
      Automatic face blurring for anonymity
    </div>
    <div class="feature">
      <strong>📊 Analytics</strong><br>
      Crowd trends, charts, and CSV exports
    </div>
    <div class="feature">
      <strong>🚨 Smart Alerts</strong><br>
      Email notifications for threshold breaches
    </div>
    <div class="feature">
      <strong>☁️ Cloud Integration</strong><br>
      AWS S3 storage with secure links
    </div>
    <div class="feature">
      <strong>🎬 Video Showcase</strong><br>
      Embedded playback of annotated results
    </div>
  </div>
</div>

<div class="box">
  <h3>🔧 System Health</h3>
  <p><a href="/test-alert" target="_blank" style="color:#667eea;text-decoration:none">🧪 Test Alert System</a> - Verify email notifications</p>
</div>

</body></html>
"""

RESULT_HTML = """
<!doctype html><html><head><title>Analysis Results</title>
<style>
 body{font-family:'Segoe UI',system-ui,sans-serif;margin:40px;max-width:1400px;background:#f8f9fa}
 .header{background:linear-gradient(135deg,#28a745 0%,#20c997 100%);color:white;padding:25px;border-radius:15px;margin-bottom:25px;text-align:center}
 .box{border:1px solid #e0e0e0;padding:25px;border-radius:15px;margin-bottom:20px;background:white;box-shadow:0 4px 6px rgba(0,0,0,0.1)}
 .video-container{position:relative;margin:20px 0;border-radius:12px;overflow:hidden;box-shadow:0 8px 16px rgba(0,0,0,0.15)}
 video{width:100%;height:auto;display:block}
 .video-overlay{position:absolute;top:15px;left:15px;background:rgba(0,0,0,0.7);color:white;padding:8px 15px;border-radius:20px;font-size:14px}
 table{border-collapse:collapse;width:100%;margin:15px 0}
 th,td{border:1px solid #ddd;padding:12px;text-align:left}
 th{background:#f8f9fa;font-weight:600}
 .stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:20px;margin:20px 0}
 .stat-card{background:linear-gradient(135deg,#f8f9fa 0%,#e9ecef 100%);padding:20px;border-radius:10px;text-align:center;border-left:4px solid #667eea}
 .stat-number{font-size:2em;font-weight:bold;color:#667eea;margin-bottom:5px}
 .stat-label{color:#666;font-size:0.9em;text-transform:uppercase;letter-spacing:1px}
 .alert-badge{background:#dc3545;color:white;padding:4px 12px;border-radius:15px;font-size:0.85em;margin-left:10px}
 .download-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:15px;margin:20px 0}
 .download-card{background:#f8f9fa;padding:15px;border-radius:10px;border-left:4px solid #28a745}
 .download-card a{color:#28a745;text-decoration:none;font-weight:500}
 .download-card a:hover{text-decoration:underline}
 .chart-container{margin:20px 0;text-align:center}
 .chart-container img{max-width:100%;border:2px solid #e0e0e0;border-radius:10px;box-shadow:0 4px 8px rgba(0,0,0,0.1)}
 .back-button{background:linear-gradient(135deg,#6c757d 0%,#495057 100%);color:white;padding:12px 25px;border:none;border-radius:8px;text-decoration:none;display:inline-block;margin-top:20px;transition:transform 0.2s}
 .back-button:hover{transform:translateY(-2px);color:white;text-decoration:none}
</style></head><body>

<div class="header">
  <h1>✅ Analysis Complete</h1>
  <p>Video processed successfully with AI detection and crowd analytics</p>
</div>

<!-- Video Showcase Section -->
<div class="box">
  <h3>🎬 Annotated Video Playback</h3>
  <p>Watch your video with real-time AI annotations showing detected people and privacy-protected faces:</p>
  
  <div class="video-container">
    <video controls preload="metadata" style="width: 100%; height: auto;">
      <source src="{{out_url}}" type="video/mp4">
      <source src="{{out_url}}" type="video/webm">
      <p style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 8px;">
        ⚠️ Video playback issue? 
        <a href="{{out_url}}" download style="color: #007bff; font-weight: bold;">Download the video file</a> 
        or try <a href="/debug/video/{{out_name}}" style="color: #007bff;">debug info</a>
      </p>
    </video>
    <div class="video-overlay">
      🤖 AI Annotated • {{max_people}} Peak People
    </div>
  </div>
  
  <div style="text-align: center; margin: 15px 0;">
    <a href="{{out_url}}" download style="background: #28a745; color: white; padding: 8px 16px; text-decoration: none; border-radius: 5px;">
      📥 Download Video
    </a>
    <a href="/debug/video/{{out_name}}" style="background: #007bff; color: white; padding: 8px 16px; text-decoration: none; border-radius: 5px; margin-left: 10px;">
      🔍 Debug Info
    </a>
  </div>
  
  <p style="text-align:center;margin-top:15px;color:#666">
    <strong>{{out_name}}</strong> • {{duration_s}}s duration • {{fps}} FPS
  </p>
</div>

<!-- Statistics Overview -->
<div class="box">
  <h3>📊 Analysis Statistics</h3>
  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-number">{{frames}}</div>
      <div class="stat-label">Total Frames</div>
    </div>
    <div class="stat-card">
      <div class="stat-number">{{max_people}}</div>
      <div class="stat-label">Peak People</div>
    </div>
    <div class="stat-card">
      <div class="stat-number">{{duration_s}}</div>
      <div class="stat-label">Duration (sec)</div>
    </div>
    <div class="stat-card">
      <div class="stat-number">{{people_thresh}}</div>
      <div class="stat-label">Alert Threshold</div>
    </div>
  </div>
</div>

<!-- Alert Status -->
<div class="box">
  <h3>🚨 Alert Status 
    {% if alerts and alerts|length>0 %}
      <span class="alert-badge">{{alerts|length}} TRIGGERED</span>
    {% else %}
      <span style="background:#28a745;color:white;padding:4px 12px;border-radius:15px;font-size:0.85em;margin-left:10px">NO ALERTS</span>
    {% endif %}
  </h3>
  
  <p><strong>Threshold Configuration:</strong> ≥{{people_thresh}} people sustained for ≥{{min_dur}} seconds</p>
  
  {% if alerts and alerts|length>0 %}
    <table>
      <thead>
        <tr><th>Alert #</th><th>Start Time</th><th>End Time</th><th>Duration</th><th>Peak People</th></tr>
      </thead>
      <tbody>
        {% for a in alerts %}
          <tr>
            <td><strong>{{loop.index}}</strong></td>
            <td>{{a.start_sec}}s</td>
            <td>{{a.end_sec}}s</td>
            <td>{{a.end_sec - a.start_sec + 1}}s</td>
            <td><strong>{{a.peak}} people</strong></td>
          </tr>
        {% endfor %}
      </tbody>
    </table>
  {% else %}
    <p style="color:#28a745;font-style:italic">✅ No crowd threshold breaches detected in this video.</p>
  {% endif %}
</div>

<!-- Downloads Section -->
<div class="box">
  <h3>📁 Download Files</h3>
  <div class="download-grid">
    <div class="download-card">
      <strong>📹 Annotated Video</strong><br>
      <a href="{{out_url}}">{{out_name}}</a>
    </div>
    <div class="download-card">
      <strong>📊 Trend Data (CSV)</strong><br>
      <a href="{{csv_url}}">{{csv_name}}</a>
    </div>
    <div class="download-card">
      <strong>📈 Trend Chart (PNG)</strong><br>
      <a href="{{png_url}}">{{png_name}}</a>
    </div>
    <div class="download-card">
      <strong>🚨 Alerts Data (JSON)</strong><br>
      <a href="{{alerts_url}}">{{alerts_name}}</a>
    </div>
  </div>

  {% if s3_vid or s3_csv or s3_png or s3_json %}
  <h4>☁️ Cloud Storage Links (Temporary)</h4>
  <div class="download-grid">
    {% if s3_vid %}<div class="download-card"><strong>S3 Video:</strong><br><a href="{{s3_vid}}">Secure Download</a></div>{% endif %}
    {% if s3_csv %}<div class="download-card"><strong>S3 CSV:</strong><br><a href="{{s3_csv}}">Secure Download</a></div>{% endif %}
    {% if s3_png %}<div class="download-card"><strong>S3 Chart:</strong><br><a href="{{s3_png}}">Secure Download</a></div>{% endif %}
    {% if s3_json %}<div class="download-card"><strong>S3 JSON:</strong><br><a href="{{s3_json}}">Secure Download</a></div>{% endif %}
  </div>
  {% endif %}
</div>

<!-- Trend Visualization -->
<div class="box">
  <h3>📈 Crowd Trend Analysis</h3>
  <div class="chart-container">
    <img src="{{png_url}}" alt="Crowd trend over time" />
  </div>
  <p style="text-align:center;color:#666;margin-top:15px">
    <small>📋 CSV contains: second, people_avg, people_max for detailed analysis</small>
  </p>
</div>

<div style="text-align:center">
  <a href="/" class="back-button">🔄 Process Another Video</a>
</div>

</body></html>
"""

# --------------------
# Small helpers (unchanged)
# --------------------
def allowed_file(name: str) -> bool:
    _, ext = os.path.splitext(name.lower())
    return ext in ALLOWED_EXT

def blur_faces(frame):
    """Blur faces for privacy using Haar cascade (CPU)."""
    if FACE_CASCADE.empty():
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30,30))
    out = frame
    for (x,y,w,h) in faces:
        roi = out[y:y+h, x:x+w]
        roi = cv2.GaussianBlur(roi, (35,35), 0)   # strong blur
        out[y:y+h, x:x+w] = roi
    return out

def compute_trend_and_alerts(per_frame_counts: List[int], fps: float,
                             people_thresh: int, min_dur: int,
                             csv_path: str, png_path: str) -> List[Dict]:
    """Write CSV & PNG of crowd trend; return alert windows where sec_max>=threshold for >=min_dur seconds."""
    counts = np.array(per_frame_counts, dtype=np.int32)
    ts = np.arange(len(counts)) / (fps or 25.0)
    secs = np.floor(ts).astype(int)
    max_sec = int(secs.max()) if len(secs) else 0

    sec_avg, sec_max = [], []
    for s in range(max_sec + 1):
        vals = counts[secs == s]
        sec_avg.append(float(vals.mean()) if vals.size else 0.0)
        sec_max.append(int(vals.max()) if vals.size else 0)

    # CSV
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["second", "people_avg", "people_max"])
        for s in range(max_sec + 1):
            wcsv.writerow([s, round(sec_avg[s],3), sec_max[s]])

    # Enhanced PNG with better styling
    plt.figure(figsize=(12,6))
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    plt.plot(range(max_sec+1), sec_avg, label="Average People/sec", linewidth=2, color='#667eea')
    plt.plot(range(max_sec+1), sec_max, label="Peak People/sec", linewidth=2, color='#764ba2')
    plt.axhline(people_thresh, linestyle="--", color='red', linewidth=2, label=f"Alert Threshold = {people_thresh}")
    plt.fill_between(range(max_sec+1), sec_max, alpha=0.3, color='#764ba2')
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Number of People", fontsize=12)
    plt.title("Crowd Size Analysis Over Time", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=120, bbox_inches='tight')
    plt.close()

    # Alerts
    alerts = []
    start = None
    for s in range(max_sec + 1):
        if sec_max[s] >= people_thresh:
            start = s if start is None else start
        else:
            if start is not None and (s - start) >= min_dur:
                peak = int(max(sec_max[start:s]))
                alerts.append({"start_sec": int(start), "end_sec": int(s-1), "peak": peak})
            start = None
    if start is not None and ((max_sec + 1) - start) >= min_dur:
        peak = int(max(sec_max[start:max_sec+1]))
        alerts.append({"start_sec": int(start), "end_sec": int(max_sec), "peak": peak})
    return alerts

def run_pipeline(in_path: str, out_path: str, csv_path: str, png_path: str, alerts_path: str,
                 conf: float = 0.35, people_thresh: int = 15, min_dur: int = 5,
                 face_blur: bool = True) -> Tuple[Dict, List[Dict]]:
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open input video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print(f"🎬 Video Info: {w}x{h} at {fps} FPS, {frames_total} frames")
    
    # Ensure we have a .mp4 extension for web compatibility
    if not out_path.endswith('.mp4'):
        out_path = out_path.rsplit('.', 1)[0] + '.mp4'
    
    # Try codecs in order of web compatibility
    codecs_to_try = [
        ('H.264 (web)', cv2.VideoWriter_fourcc(*"mp4v")),  # Most compatible
        ('H.264 alt', cv2.VideoWriter_fourcc(*"avc1")),
        ('XVID', cv2.VideoWriter_fourcc(*"XVID")),
        ('MJPG', cv2.VideoWriter_fourcc(*"MJPG"))
    ]
    
    writer = None
    for codec_name, fourcc in codecs_to_try:
        print(f"🎬 Trying {codec_name} codec...")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if writer.isOpened():
            print(f"🎬 ✅ Success with {codec_name} codec!")
            break
        else:
            if writer:
                writer.release()
            writer = None
    
    if writer is None:
        # Last resort - try without fourcc
        print("🎬 Trying default codec...")
        writer = cv2.VideoWriter(out_path, -1, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError("Could not initialize video writer with any codec")
        
    print(f"🎬 Output video writer initialized: {out_path}")

    proc_frames = 0
    max_people = 0
    per_frame_counts: List[int] = []

    print(f"Processing video: {frames_total} frames at {fps} FPS...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if face_blur:
            frame = blur_faces(frame)

        res = model.predict(frame, conf=conf, verbose=False)[0]
        n_people = sum(int(b.cls) == 0 for b in res.boxes)
        max_people = max(max_people, n_people)
        per_frame_counts.append(n_people)

        annotated = res.plot()
        writer.write(annotated)
        proc_frames += 1
        
        if proc_frames % 30 == 0:  # Progress indicator
            print(f"Processed {proc_frames}/{frames_total} frames...")

    cap.release()
    writer.release()

    alerts = compute_trend_and_alerts(per_frame_counts, fps, people_thresh, min_dur, csv_path, png_path)

    with open(alerts_path, "w") as f:
        json.dump({
            "people_threshold": people_thresh,
            "min_duration_seconds": min_dur,
            "max_people_detected": int(max_people),
            "total_frames": proc_frames,
            "alerts": alerts
        }, f, indent=2)

    duration_s = round(proc_frames / (fps or 25.0), 2)
    stats = {
        "frames": frames_total or proc_frames,
        "proc_frames": proc_frames,
        "fps": round(fps, 2),
        "duration_s": duration_s,
        "max_people": int(max_people),
    }
    return stats, alerts

def send_alert_emails(alerts: List[Dict], people_thresh: int, min_dur: int,
                      out_name: str, csv_name: str, png_name: str,
                      out_url: str = "", csv_url: str = "", png_url: str = "", json_url: str = ""):
    """Send SNS email to moldplay267@gmail.com with alert details and links."""
    if not alerts:
        return
    
    # Enhanced email with more details
    windows = ", ".join([f"{a['start_sec']}-{a['end_sec']}s ({a['peak']} people)" for a in alerts])
    peak_all = max([a["peak"] for a in alerts]) if alerts else 0
    total_duration = sum([a["end_sec"] - a["start_sec"] + 1 for a in alerts])
    
    subject = f"🚨 CAMPUS ALERT: {len(alerts)} Crowd Event(s) Detected - Peak {peak_all} People"
    
    lines = [
        "🎯 CAMPUS AI SURVEILLANCE SYSTEM ALERT",
        "=" * 50,
        f"📊 DETECTION SUMMARY:",
        f"   • Alert Threshold: ≥{people_thresh} people for ≥{min_dur}s",
        f"   • Events Detected: {len(alerts)}",
        f"   • Peak Crowd Size: {peak_all} people",
        f"   • Total Alert Duration: {total_duration} seconds",
        "",
        f"⏰ EVENT TIMELINE:",
        f"   • Time Windows: {windows}",
        "",
        f"📁 ANALYSIS FILES:",
        f"   • Annotated Video: {out_name}",
        f"   • Crowd Data (CSV): {csv_name}",
        f"   • Trend Chart (PNG): {png_name}",
        "",
        f"🔗 SECURE DOWNLOAD LINKS:",
    ]
    
    if out_url:  lines.append(f"   📹 Video: {out_url}")
    if csv_url:  lines.append(f"   📊 Data:  {csv_url}")
    if png_url:  lines.append(f"   📈 Chart: {png_url}")
    if json_url: lines.append(f"   📋 JSON:  {json_url}")
    
    lines.extend([
        "",
        "⚠️  Links expire in 1 hour for security.",
        "📧 This alert was sent to: moldplay267@gmail.com",
        "",
        "🤖 Automated Campus AI Detection System"
    ])
    
    ok, info = publish_sns(subject, "\n".join(lines))
    print(f"SNS Alert Email: {ok} - {info}")

# --------------------
# Routes
# --------------------
@app.get("/")
def index():
    return render_template_string(INDEX_HTML, max_mb=MAX_MB)

@app.get("/outputs/<path:fname>")
def download(fname):
    """Serve files from outputs directory with proper headers"""
    # Clean the filename to prevent path traversal
    clean_fname = os.path.basename(fname)
    path = os.path.join(OUTPUT_DIR, clean_fname)
    
    if not os.path.isfile(path):
        print(f"❌ File not found: {path}")
        abort(404, f"File not found: {clean_fname}")
    
    file_size = os.path.getsize(path)
    print(f"📁 Serving file: {clean_fname} (size: {file_size:,} bytes)")
    
    try:
        # For video files, use streaming response
        if clean_fname.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            return send_from_directory(
                OUTPUT_DIR, 
                clean_fname,
                mimetype='video/mp4',
                as_attachment=False,
                conditional=True
            )
        else:
            # Regular files
            return send_from_directory(OUTPUT_DIR, clean_fname)
            
    except Exception as e:
        print(f"❌ Error serving file: {str(e)}")
        abort(500, f"Error serving file: {str(e)}")

@app.get("/test-alert")
def test_alert():
    # First, let's check what SNS topics exist
    topic_list = "Could not list topics"
    if sns:
        try:
            response = sns.list_topics()
            topics = response.get('Topics', [])
            if topics:
                topic_list = "\n".join([f"   • {t['TopicArn']}" for t in topics])
            else:
                topic_list = "   No topics found"
        except Exception as e:
            topic_list = f"   Error: {str(e)}"
    
    # Now test the alert
    test_subject = "🧪 TEST: Campus AI System Check"
    test_message = """🎯 CAMPUS AI SURVEILLANCE SYSTEM TEST
================================================
📊 SYSTEM STATUS: ✅ OPERATIONAL

This is a test alert to verify email notifications.

📧 Recipient: moldplay267@gmail.com
🕒 Test Time: System operational check
🤖 Automated Campus AI Detection System

If you receive this message, alerts are working correctly!
"""
    ok, info = publish_sns(test_subject, test_message)
    
    status_msg = f"""
📧 Email Alert Test Results
{'='*50}
Status: {'✅ SUCCESS' if ok else '❌ FAILED'}
Details: {info}
Target: moldplay267@gmail.com
Region: us-east-2

🎯 CONFIGURED TOPIC ARN:
{SNS_TOPIC_ARN}

📋 AVAILABLE TOPICS IN YOUR ACCOUNT:
{topic_list}

💡 TROUBLESHOOTING:
{'Email sent successfully!' if ok else 'Check if the topic ARN matches an available topic above.'}
"""
    
    return (status_msg, 200) if ok else (status_msg, 500)

@app.post("/process")
def process():
    if "video" not in request.files:
        abort(400, "No file part")
    f = request.files["video"]
    if f.filename == "":
        abort(400, "No selected file")
    if not allowed_file(f.filename):
        abort(400, f"Unsupported extension. Allowed: {', '.join(sorted(ALLOWED_EXT))}")

    conf = max(0.1, min(0.7, request.form.get("conf", type=float, default=0.35)))
    people_thresh = request.form.get("people_thresh", type=int, default=15)
    min_dur = request.form.get("min_dur", type=int, default=5)
    use_faceblur = bool(request.form.get("faceblur"))

    uid = uuid.uuid4().hex[:8]  # Shorter UID
    # Clean filename - remove special characters and limit length
    clean_filename = "".join(c for c in f.filename if c.isalnum() or c in ".-_")[:50]
    if '.' in clean_filename:
        name_part, ext_part = clean_filename.rsplit('.', 1)
        in_name = f"{uid}_{name_part[:30]}.{ext_part}"
    else:
        in_name = f"{uid}_{clean_filename[:30]}.mp4"
    
    in_path = os.path.join(UPLOAD_DIR, in_name)
    f.save(in_path)

    # Create clean output filenames without duplicate suffixes
    base_name = f"{uid}_video"
    out_name = base_name + "_annotated.mp4"
    csv_name = base_name + "_trend.csv"
    png_name = base_name + "_trend.png"
    alerts_name = base_name + "_alerts.json"

    out_path = os.path.join(OUTPUT_DIR, out_name)
    csv_path = os.path.join(OUTPUT_DIR, csv_name)
    png_path = os.path.join(OUTPUT_DIR, png_name)
    alerts_path = os.path.join(OUTPUT_DIR, alerts_name)

    print(f"Starting video processing: {in_name}")
    
    try:
        stats, alerts = run_pipeline(
            in_path, out_path, csv_path, png_path, alerts_path,
            conf=conf, people_thresh=people_thresh, min_dur=min_dur,
            face_blur=use_faceblur
        )
        print(f"Processing complete. Found {len(alerts)} alerts.")
        
        # Verify all files were created
        missing_files = []
        for file_path, name in [(out_path, "video"), (csv_path, "CSV"), (png_path, "PNG"), (alerts_path, "JSON")]:
            if not os.path.exists(file_path):
                missing_files.append(name)
            else:
                size = os.path.getsize(file_path)
                print(f"✅ Created {name}: {os.path.basename(file_path)} ({size:,} bytes)")
        
        if missing_files:
            raise RuntimeError(f"Missing output files: {', '.join(missing_files)}")
            
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        abort(500, f"Video processing failed: {str(e)}")

    # Upload artifacts to S3 + pre-sign
    run_id  = uuid.uuid4().hex[:12]
    s3_vid  = upload_and_sign(out_path,    run_id)
    s3_csv  = upload_and_sign(csv_path,    run_id)
    s3_png  = upload_and_sign(png_path,    run_id)
    s3_json = upload_and_sign(alerts_path, run_id)

    # Send alert email if triggered
    if alerts:
        send_alert_emails(
            alerts=alerts, people_thresh=people_thresh, min_dur=min_dur,
            out_name=os.path.basename(out_path), csv_name=os.path.basename(csv_path),
            png_name=os.path.basename(png_path),
            out_url=s3_vid, csv_url=s3_csv, png_url=s3_png, json_url=s3_json
        )

    return render_template_string(
        RESULT_HTML,
        in_name=os.path.basename(in_path),
        out_name=os.path.basename(out_path), out_url=f"/outputs/{os.path.basename(out_path)}",
        csv_name=os.path.basename(csv_path), csv_url=f"/outputs/{os.path.basename(csv_path)}",
        png_name=os.path.basename(png_path), png_url=f"/outputs/{os.path.basename(png_path)}",
        alerts_name=os.path.basename(alerts_path), alerts_url=f"/outputs/{os.path.basename(alerts_path)}",
        frames=stats["frames"], proc_frames=stats["proc_frames"], fps=stats["fps"],
        duration_s=stats["duration_s"], max_people=stats["max_people"],
        people_thresh=people_thresh, min_dur=min_dur, alerts=alerts,
        s3_vid=s3_vid, s3_csv=s3_csv, s3_png=s3_png, s3_json=s3_json
    )

# --------------------
# Additional utility routes
# --------------------
@app.get("/health")
def health_check():
    """System health check endpoint"""
    health_info = {
        "status": "healthy",
        "yolo_model": "yolov8n.pt loaded" if model else "not loaded",
        "face_cascade": "loaded" if not FACE_CASCADE.empty() else "not loaded",
        "sns_configured": bool(sns and SNS_TOPIC_ARN),
        "s3_configured": bool(s3 and S3_BUCKET),
        "upload_dir": os.path.exists(UPLOAD_DIR),
        "output_dir": os.path.exists(OUTPUT_DIR)
    }
    
    # Test AWS credentials
    aws_test = "❌ Not tested"
    if sns:
        try:
            # Try to list topics to test credentials
            response = sns.list_topics()
            aws_test = "✅ Credentials working"
        except Exception as e:
            aws_test = f"❌ Credential error: {str(e)[:50]}..."
    
    return f"""
🏥 SYSTEM HEALTH CHECK
{'='*40}
🤖 Status: {health_info['status'].upper()}
📊 YOLO Model: {health_info['yolo_model']}
👤 Face Detection: {health_info['face_cascade']}
📧 SNS Alerts: {'✅ Ready' if health_info['sns_configured'] else '❌ Not configured'}
🔑 AWS Credentials: {aws_test}
☁️  S3 Storage: {'✅ Ready' if health_info['s3_configured'] else '❌ Not configured'}
📁 Upload Dir: {'✅ Ready' if health_info['upload_dir'] else '❌ Missing'}
📁 Output Dir: {'✅ Ready' if health_info['output_dir'] else '❌ Missing'}

📧 Target Email: moldplay267@gmail.com
🎯 SNS Topic: aidetect
🌍 Region: us-east-2
📋 Topic ARN: {SNS_TOPIC_ARN[:50]}...

🔧 Debug URLs:
   • /test-alert - Test email alerts
   • /debug/video/filename.mp4 - Check video properties
""", 200

@app.get("/debug/files")
def list_output_files():
    """List all files in the output directory with clean display"""
    try:
        files = []
        if os.path.exists(OUTPUT_DIR):
            for f in os.listdir(OUTPUT_DIR):
                path = os.path.join(OUTPUT_DIR, f)
                if os.path.isfile(path):
                    size = os.path.getsize(path)
                    size_mb = size / (1024 * 1024)
                    
                    # Show file type icon
                    if f.endswith('.mp4'):
                        icon = "🎬"
                    elif f.endswith('.csv'):
                        icon = "📊"
                    elif f.endswith('.png'):
                        icon = "📈"
                    elif f.endswith('.json'):
                        icon = "📋"
                    else:
                        icon = "📁"
                    
                    files.append(f"{icon} <a href='/outputs/{f}'>{f}</a> ({size_mb:.1f} MB)")
        
        files_list = "<br>".join(files) if files else "No files found"
        
        return f"""
<!DOCTYPE html>
<html><head><title>Output Files</title>
<style>body{{font-family:system-ui;margin:40px;max-width:800px}}
.clean{{background:#e8f5e8;padding:10px;border-radius:5px;margin:10px 0}}
</style></head><body>

<h2>📂 Output Directory Contents</h2>
<p><strong>Directory:</strong> {OUTPUT_DIR}</p>
<p><strong>Total files:</strong> {len(files)}</p>

<div class="clean">
<strong>🧹 Clean Filenames:</strong> New uploads will now use clean, short filenames like:<br>
• <code>a1b2c3d4_video_annotated.mp4</code><br>
• <code>a1b2c3d4_video_trend.csv</code><br>
• <code>a1b2c3d4_video_trend.png</code>
</div>

<h3>📁 Current Files:</h3>
{files_list}

<p><strong>🔧 Actions:</strong>
<a href="/debug/cleanup">Clean old files</a> | 
<a href="/">Upload new video</a>
</p>

</body></html>
        """, 200
    except Exception as e:
        return f"❌ Error listing files: {str(e)}", 500

@app.get("/debug/cleanup")
def cleanup_files():
    """Clean up old files with problematic names"""
    try:
        cleaned = []
        if os.path.exists(OUTPUT_DIR):
            for f in os.listdir(OUTPUT_DIR):
                # Remove files with very long names or multiple _annotated
                if len(f) > 100 or "_annotated_annotated" in f:
                    path = os.path.join(OUTPUT_DIR, f)
                    os.remove(path)
                    cleaned.append(f)
        
        return f"""
🧹 CLEANUP COMPLETE
{'='*50}
Removed {len(cleaned)} problematic files:
{chr(10).join(cleaned) if cleaned else 'No files needed cleaning'}

<a href="/debug/files">View remaining files</a> | <a href="/">Upload new video</a>
        """, 200
    except Exception as e:
        return f"❌ Cleanup error: {str(e)}", 500
def debug_video(filename):
    """Debug video file properties"""
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(path):
        return f"File not found: {filename}", 404
    
    # Get file info
    file_size = os.path.getsize(path)
    
    # Try to read video properties
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        cap.release()
        
        info = f"""
🎬 VIDEO DEBUG INFO: {filename}
{'='*50}
📁 File Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)
📺 Dimensions: {width}x{height}
🎯 Frame Rate: {fps} FPS
🎞️  Frame Count: {frame_count}
⏱️  Duration: {frame_count/fps:.1f} seconds
🔧 Codec: {codec} ({fourcc})
📂 Path: {path}
✅ OpenCV can read: Yes

🌐 Browser Compatibility:
   - H.264/MP4: {'✅ Good' if codec in ['avc1', 'h264'] else '⚠️ May not work'}
   - Size: {'✅ Good' if file_size < 100*1024*1024 else '⚠️ Large file'}
"""
    else:
        info = f"""
❌ VIDEO DEBUG ERROR: {filename}
{'='*50}
📁 File Size: {file_size:,} bytes
📂 Path: {path}
❌ OpenCV cannot read this file
❌ Possible codec issue
"""
    
    return info, 200
def too_large(e):
    return f"File too large. Maximum size: {MAX_MB}MB", 413

@app.errorhandler(400)
def bad_request(e):
    return f"Bad request: {e.description}", 400

@app.errorhandler(500)
def internal_error(e):
    return f"Internal error: {e.description}", 500

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    print("🚀 Starting Campus Protest Detection System...")
    print(f"📧 Email alerts configured for: moldplay267@gmail.com")
    print(f"🌐 SNS Topic: aidetect (us-east-2)")
    print(f"📊 YOLO Model: YOLOv8n (CPU optimized)")
    print(f"🔒 Face blurring: OpenCV Haar Cascades")
    print(f"📁 Upload limit: {MAX_MB}MB")
    print(f"🎯 Supported formats: {', '.join(sorted(ALLOWED_EXT))}")
    print("="*50)
    
    # Run development server
    app.run(host="0.0.0.0", port=5000, debug=True)