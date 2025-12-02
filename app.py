

# .\venv\Scripts\Activate.ps1   venv\Scripts\activate
# python app.py

from flask import Flask, request, jsonify, send_from_directory, render_template ,session, redirect, url_for
import mysql.connector
from functools import wraps
from dotenv import load_dotenv
import os
import json
import datetime
import uuid
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import mediapipe as mp
import numpy as np
import base64
import requests
import traceback
from flask_socketio import SocketIO, emit, join_room, leave_room
from engineio.async_drivers import threading






# ===========================
# Connected users storage
# ===========================
connected_users = {}   # { user_id: socket_id }
active_calls = {}      # { call_id: admin_sid }

# 🤖 Import intelligent analysis system
from ml_models.posture_analyzer import PostureAnalyzer

# --- MediaPipe initialization ---
mp_pose = mp.solutions.pose
pose_3d = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# 🤖 Initialize intelligent analyzer
print("🤖 Loading intelligent posture analysis system...")
posture_analyzer = PostureAnalyzer()
print("✅ Intelligent analysis system ready!")

load_dotenv()
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
app.secret_key = "super_secret_key_for_admin_dashboard_123" 
app.config['SECRET_KEY'] = 'fd345@#$vd_8934_secure_key_pose_app_2025'


@app.route('/')
def home():
    return "✅ Server is Running Successfully! (Pose API)"




# --- Database connection setup ---
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DB = os.getenv("MYSQL_DB")
db_connection = None

# --- Upload folder setup ---
UPLOAD_FOLDER = 'static/uploads'


def create_db_connection():
    try:
        # ✅ قراءة البورت من Environment Variables (ضروري لـ Aiven)
        # إلا مالقاهش كيستعمل 3306
        db_port = int(os.getenv("MYSQL_PORT", 3306))
        
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            port=db_port  # ⬅️ هادي هي اللي كانت ناقصة
        )
        print("✅ Connected to MySQL successfully!")
        return conn
    except Exception as e:
        print(f"❌ Error connecting to MySQL: {e}")
        return None


def get_db_connection():
    """
    Reuse global connection if possible, otherwise create a new one.
    """
    global db_connection
    try:
        if db_connection and hasattr(db_connection, "is_connected") and db_connection.is_connected():
            return db_connection
    except Exception:
        pass

    db_connection = create_db_connection()
    return db_connection


def ensure_tables_exist(conn):
    if not conn:
        return
    try:
        cursor = conn.cursor()

        # --- users table ---
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            email VARCHAR(255) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            birth_date DATE NOT NULL,
            role VARCHAR(50) DEFAULT 'user'
        );
        """)

        # ensure role column
        cursor.execute("""
        SELECT COUNT(*)
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s
        AND TABLE_NAME = 'users'
        AND COLUMN_NAME = 'role'
        """, (MYSQL_DB,))
        role_exists = cursor.fetchone()[0]
        if role_exists == 0:
            cursor.execute("""
            ALTER TABLE users
            ADD COLUMN role VARCHAR(50) DEFAULT 'user'
            """)
            print("✅ Added 'role' column to users table")

        # --- poses table ---
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS poses (
            id VARCHAR(255) PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            timestamp DATETIME,
            image_path VARCHAR(1024),
            shoulder_y_diff DECIMAL(10, 5),
            hip_y_diff DECIMAL(10, 5),
            shoulder_higher_side VARCHAR(255),
            hip_higher_side VARCHAR(255),
            landmarks_3d_json JSON,
            poses_json JSON,
            analysis_json JSON,
            admin_comment TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """)

        # --- call_logs table (MySQL ONLY, no SQLite) ---
       # --- call_logs table (نسخة بسيطة ومتوافقة) ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS call_logs (
                id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                doctor_id VARCHAR(255) DEFAULT NULL,
                started_at DATETIME NOT NULL,
                ended_at DATETIME DEFAULT NULL,
                duration_seconds INT DEFAULT NULL,
                call_type VARCHAR(50) DEFAULT 'video',
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        conn.commit()
        cursor.close()
        print("✅ Tables ensured (users, poses, call_logs).")
    except Exception as e:
        print(f"❌ Error ensuring tables exist: {e}")


def ensure_admin_exists(conn):
    if not conn:
        return
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE role = 'admin' LIMIT 1;")
        admin = cursor.fetchone()

        if not admin:
            admin_id = str(uuid.uuid4())
            password_hash = generate_password_hash("admin123")

            cursor.execute("""
                INSERT INTO users (id, name, email, password_hash, birth_date, role)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (admin_id, "Admin User", "admin@posture.com", password_hash, "1990-01-01", "admin"))
            conn.commit()
            print("✅ Default admin created: admin@posture.com / admin123")
        else:
            print("ℹ️ Admin already exists.")
        cursor.close()
    except Exception as e:
        print(f"❌ Error ensuring admin exists: {e}")


# --- Run setup ---
db_connection = create_db_connection()
if db_connection:
    ensure_tables_exist(db_connection)
    ensure_admin_exists(db_connection)

# --- Create uploads directory if not exists ---
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"✅ Created upload folder: {UPLOAD_FOLDER}")








# دالة مساعدة لتحويل الاسم من LEFT_SHOULDER إلى leftShoulder
def to_camel_case(snake_str):
    components = snake_str.lower().split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

@app.route('/save-snapshot-analysis', methods=['POST'])
def save_snapshot_analysis():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        image_base64_data = data.get('image_base64')

        if not user_id or not image_base64_data:
            return jsonify({"error": "Missing user_id or image data"}), 400

        # 1. فك تشفير الصورة
        if ',' in image_base64_data:
            image_base64_data = image_base64_data.split(',')[1]
        image_data = base64.b64decode(image_base64_data)
        np_arr = np.frombuffer(image_data, np.uint8)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # استخراج الأبعاد (مهم جداً)
        height, width, _ = image_bgr.shape
        print(f"📏 Snapshot Dimensions: {width}x{height}")

        timestamp = datetime.datetime.now()
        filename = f"snap_{user_id[:8]}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        cv2.imwrite(filepath, image_bgr)
        
        db_image_path = f"/uploads/{filename}"

        # 2. تحليل الصورة بـ MediaPipe
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        analysis_result = {}
        poses_data = [] 

        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose_processor:
            results = pose_processor.process(image_rgb)
            
            if results.pose_landmarks: # نستعمل pose_landmarks (Normalized) للرسم 2D
                landmarks_map = {}
                
                # تحويل Landmarks إلى الصيغة اللي كيفهمها Flutter
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    # MediaPipe كيعطينا: LEFT_SHOULDER
                    raw_name = mp_pose.PoseLandmark(i).name
                    # Flutter باغي: leftShoulder
                    name_camel = to_camel_case(raw_name)
                    
                    landmarks_map[name_camel] = {
                        "x": landmark.x * width,   # تحويل من 0..1 إلى Pixels
                        "y": landmark.y * height,
                        "z": landmark.z * width,   # تقريب للعمق
                        "likelihood": landmark.visibility
                    }
                
                # ✅ الهيكل الصحيح: poses عبارة عن List فيها Object فيه landmarks كـ Map
                poses_data = [{"landmarks": landmarks_map}]

                # التحليل (Analysis)
                try:
                    # نستعمل landmarks 3D للتحليل الدقيق
                    lm_3d = results.pose_world_landmarks.landmark
                    
                    ls_y = lm_3d[11].y
                    rs_y = lm_3d[12].y
                    shoulder_diff = abs(ls_y - rs_y) * 100 

                    lh_y = lm_3d[23].y
                    rh_y = lm_3d[24].y
                    hip_diff = abs(lh_y - rh_y) * 100

                    analysis_result = {
                        "shoulder_y_diff": shoulder_diff,
                        "hip_y_diff": hip_diff,
                        "shoulder_higher_side": "Right" if rs_y < ls_y else "Left",
                        "hip_higher_side": "Right" if rh_y < lh_y else "Left",
                        
                        # ✅ إضافة الأبعاد باش Flutter يخدم مرتاح
                        "image_dimensions": {
                            "width": width,
                            "height": height
                        }
                    }
                except Exception as analysis_err:
                    print(f"⚠️ Analysis calc error: {analysis_err}")
                    analysis_result = {
                        "error": "Calculation failed",
                        "image_dimensions": {"width": width, "height": height}
                    }
            else:
                analysis_result = {
                    "shoulder_y_diff": 0, "hip_y_diff": 0,
                    "image_dimensions": {"width": width, "height": height}
                }

        # 3. التسجيل في MySQL
        record_id = str(uuid.uuid4())
        
        sh_diff = analysis_result.get('shoulder_y_diff', 0)
        hp_diff = analysis_result.get('hip_y_diff', 0)
        sh_side = analysis_result.get('shoulder_higher_side', 'N/A')
        hp_side = analysis_result.get('hip_higher_side', 'N/A')

        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
            INSERT INTO poses (
                id, user_id, timestamp, image_path,
                shoulder_y_diff, hip_y_diff,
                shoulder_higher_side, hip_higher_side,
                poses_json, analysis_json, admin_comment
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            record_id, user_id, timestamp, db_image_path,
            sh_diff, hp_diff, sh_side, hp_side,
            json.dumps(poses_data), json.dumps(analysis_result), 
            "Snapshot taken by Doctor during video call"
        )
        
        cursor.execute(query, values)
        conn.commit()
        cursor.close()

        print(f"✅ Snapshot saved to DB for user {user_id}")
        return jsonify({"success": True, "record_id": record_id}), 200

    except Exception as e:
        print(f"❌ Error saving snapshot: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
# ============================================================
# 🔔 CALL / VIDEO LOGIC — MySQL ONLY
# ============================================================




# في app.py - زيد هاد الـ route

@app.route('/admin/video_call/<call_id>')
def admin_video_call(call_id):
    # ✅ الطريقة الصحيحة: Flask كيمشي نيشان لمجلد templates
    # request.host_url كتعطيك الرابط ديال السيرفر أوتوماتيك (سواء localhost أو IP)
    return render_template('admin_video_call.html', call_id=call_id, socket_url=request.host_url)





@app.route('/api/calls/ring', methods=['POST'])
def ring_user():
    data = request.get_json() or {}
    user_id = data.get('user_id')

    if not user_id:
        return jsonify({"success": False, "message": "user_id is required"}), 400

    # مجرد إشعار: الطبيب بغا يهضر مع هاد اليوزر
    socketio.emit('incoming_call', {
        'user_id': user_id,
        'message': 'Incoming call from doctor'
    }, broadcast=True)

    return jsonify({"success": True})


@app.route('/api/calls/start', methods=['POST'])
def start_call():
    """
    Doctor starts a call -> create row in call_logs (MySQL)
    and emit incoming_call with call_id to Flutter.
    """
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id')
        doctor_id = data.get('doctor_id', None)

        if not user_id:
            return jsonify({"success": False, "message": "user_id is required"}), 400

        conn = get_db_connection()
        if not conn:
            return jsonify({"success": False, "message": "DB connection failed"}), 500

        cur = conn.cursor()
        cur.execute("""
            INSERT INTO call_logs (user_id, doctor_id, started_at, call_type)
            VALUES (%s, %s, NOW(), %s)
        """, (user_id, doctor_id, "video"))
        conn.commit()

        call_id = cur.lastrowid  # INT AUTO_INCREMENT
        cur.close()

        print(f"📞 Starting call for user: {user_id} with Call ID: {call_id}")

        # Emit to Flutter
        socketio.emit('incoming_call', {
            "user_id": user_id,
            "call_id": str(call_id),
            "caller_name": "Dr. Physiotherapist",
            "call_type": "video"
        }, broadcast=True)

        return jsonify({
            "success": True,
            "message": "Call started successfully",
            "call_id": call_id
        }), 200

    except Exception as e:
        print(f"❌ Error starting call: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/calls/end', methods=['POST'])
def end_call():
    """
    Flutter أو الطبيب يرسل call_id → نحسب المدة ونسد المكالمة فـ MySQL
    """
    try:
        data = request.get_json() or {}
        call_id = data.get('call_id')

        if not call_id:
            return jsonify({"success": False, "message": "call_id is required"}), 400

        conn = get_db_connection()
        if not conn:
            return jsonify({"success": False, "message": "DB connection failed"}), 500

        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT started_at FROM call_logs WHERE id = %s", (call_id,))
        row = cur.fetchone()

        if not row:
            cur.close()
            return jsonify({"success": False, "message": "Call not found"}), 404

        started_at = row['started_at']
        ended_at = datetime.datetime.utcnow()
        duration_seconds = int((ended_at - started_at).total_seconds())

        cur.execute("""
            UPDATE call_logs
            SET ended_at = %s, duration_seconds = %s
            WHERE id = %s
        """, (ended_at, duration_seconds, call_id))
        conn.commit()
        cur.close()

        print(f"📴 Call {call_id} ended. Duration: {duration_seconds} seconds")

        return jsonify({
            "success": True,
            "ended_at": ended_at.isoformat() + "Z",
            "duration_seconds": duration_seconds
        })

    except Exception as e:
        print(f"❌ Error ending call: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500



@socketio.on("register_user")
def register_user(data):
    user_id = data.get("user_id")
    sid = request.sid

    if user_id:
        connected_users[user_id] = sid
        print(f"🔌 Registered user → {user_id} = {sid}")



@app.route('/admin/get-last-call/<user_id>')
def get_last_call(user_id):
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "message": "DB error"}), 500

    cur = conn.cursor(dictionary=True)
    # نجيبو آخر مكالمة (تكون مسجلة وفيها مدة زمنية)
    cur.execute("""
        SELECT * FROM call_logs 
        WHERE user_id = %s AND duration_seconds IS NOT NULL
        ORDER BY started_at DESC LIMIT 1
    """, (user_id,))
    row = cur.fetchone()
    cur.close()

    if not row:
        return jsonify({"success": True, "has_call": False})

    return jsonify({
        "success": True,
        "has_call": True,
        "call": {
            "started_at": row["started_at"].isoformat() if row["started_at"] else None,
            "duration_seconds": row["duration_seconds"]
        }
    })


@app.route('/admin/get-call-history/<user_id>', methods=['GET'])
def get_call_history(user_id):
    """
    History كامل ديال المكالمات لهذا اليوزر (MySQL فقط)
    """
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"success": False, "message": "DB connection failed"}), 500

        cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT *
            FROM call_logs
            WHERE user_id = %s
            ORDER BY started_at DESC
        """, (user_id,))
        rows = cur.fetchall()
        cur.close()

        for r in rows:
            if r["started_at"]:
                r["started_at"] = r["started_at"].isoformat()
            if r["ended_at"]:
                r["ended_at"] = r["ended_at"].isoformat()

        return jsonify({
            "success": True,
            "user_id": user_id,
            "total_calls": len(rows),
            "calls": rows
        })
    except Exception as e:
        print("Error get_call_history:", e)
        traceback.print_exc()
        return jsonify({"success": False, "error": "Server error"}), 500




##################################

# ===== WebRTC Signaling =====

# في app.py - Socket.IO handlers


# قاموس لتتبع المكالمات النشطة
active_calls = {}

# ===========================
# ✅ FIXED: Socket Call Logic with DB Recording
# ===========================

@socketio.on('start_call')
def handle_start_call(data):
    user_id = data.get('user_id')
    call_id = str(uuid.uuid4())
    
    print(f"\n{'='*60}")
    print(f"📞 NEW CALL REQUEST: {call_id}")
    
    # 1. محاولة التسجيل باستعمال اتصال جديد ومستقل (لضمان العملية)
    try:
        # نصاوبو كونيكسيون جديدة خاصة بهاد العملية
        new_conn = mysql.connector.connect(
            host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD, database=MYSQL_DB
        )
        if new_conn.is_connected():
            cur = new_conn.cursor()
            print(f"🛠️ DB Connected. Inserting call log...")
            
            # تنفيذ الاستعلام
            sql = "INSERT INTO call_logs (id, user_id, started_at, call_type) VALUES (%s, %s, %s, %s)"
            val = (call_id, user_id, datetime.datetime.now(), 'video')
            
            cur.execute(sql, val)
            new_conn.commit() # تأكيد الحفظ
            
            print(f"✅✅ SUCCESS: Call {call_id} inserted into DB! Rows affected: {cur.rowcount}")
            cur.close()
            new_conn.close()
        else:
            print("❌ DB Connection failed inside start_call")
            
    except mysql.connector.Error as err:
        print(f"❌❌ MYSQL ERROR: {err}")
    except Exception as e:
        print(f"❌❌ GENERAL ERROR: {e}")
        traceback.print_exc()

    # 2. إكمال عملية الاتصال
    active_calls[call_id] = {
        'admin_sid': request.sid,
        'user_id': user_id,
        'status': 'calling'
    }
    
    join_room(call_id)
    join_room(user_id)
    
    socketio.emit('incoming_call', {
        'call_id': call_id,
        'user_id': user_id,
        'user_name': 'Doctor Admin',
        'from': 'admin'
    }, room=user_id, namespace='/')
    
    return {'success': True, 'call_id': call_id}


@socketio.on('end_call')
def handle_end_call(data):
    call_id = data.get('call_id')
    print(f"📞 Ending call request: {call_id}")
    
    # 1. تحديث قاعدة البيانات
    try:
        new_conn = mysql.connector.connect(
            host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD, database=MYSQL_DB
        )
        if new_conn.is_connected():
            cur = new_conn.cursor(dictionary=True)
            
            # جلب وقت البداية
            cur.execute("SELECT started_at FROM call_logs WHERE id = %s", (call_id,))
            row = cur.fetchone()
            
            if row:
                started_at = row['started_at']
                ended_at = datetime.datetime.now()
                duration = int((ended_at - started_at).total_seconds())
                
                # تحديث النهاية
                update_sql = "UPDATE call_logs SET ended_at = %s, duration_seconds = %s WHERE id = %s"
                cur.execute(update_sql, (ended_at, duration, call_id))
                new_conn.commit()
                print(f"✅✅ SUCCESS: Call updated. Duration: {duration}s")
            else:
                print(f"⚠️ Warning: Call ID {call_id} not found in DB to update.")
                
            cur.close()
            new_conn.close()
            
    except Exception as e:
        print(f"❌ DB Update Error: {e}")

    # 2. تنظيف الذاكرة
    if call_id in active_calls:
        emit('call_ended', {'call_id': call_id}, room=call_id, namespace='/', include_self=False)
        del active_calls[call_id]




@socketio.on('call_accepted')
def handle_call_accepted(data):
    """
    Patient قبل المكالمة
    """
    call_id = data.get('call_id')
    user_id = data.get('user_id')  # ✅ Add user_id from Flutter
    
    if call_id in active_calls:
        active_calls[call_id]['status'] = 'active'
        
        # ✅ Patient joins room with call_id
        join_room(call_id)
        
        # إعلام الـ admin
        admin_sid = active_calls[call_id].get('admin_sid')
        if admin_sid:
            emit('call_accepted', {
                'call_id': call_id,
                'user_id': active_calls[call_id]['user_id']
            }, room=admin_sid, namespace='/')
        
        print(f"✅ Call {call_id} accepted by user {user_id}")

# 1. بدّل الدالة القديمة handle_webrtc_offer بهادي:
@socketio.on('webrtc_offer')
def handle_webrtc_offer(data):
    """
    ✅ FIXED: تخزين Offer إذا كان Admin في مرحلة الانتقال بين الصفحات
    """
    call_id = data.get('call_id')
    user_id = data.get('user_id')
    offer = data.get('offer')
    
    print(f"📩 Received offer for call {call_id} from user {user_id}")
    
    if call_id in active_calls:
        # ✅ تخزين الـ Offer في الذاكرة (هذا هو الحل السحري)
        active_calls[call_id]['offer'] = offer
        print(f"💾 Offer CACHED for call {call_id} (Waiting for Admin to rejoin)")

        # نحاول نصيفطوه إيلا كان Admin ديجا كاين
        admin_sid = active_calls[call_id].get('admin_sid')
        
        # نرسل للـ Room احتياطاً
        emit('incoming_webrtc_offer', {
            'call_id': call_id,
            'user_id': user_id,
            'offer': offer
        }, room=call_id, namespace='/')
        
        # نرسل للـ Admin مباشرة إذا كان متصل
        if admin_sid:
            emit('incoming_webrtc_offer', {
                'call_id': call_id,
                'user_id': user_id,
                'offer': offer
            }, room=admin_sid, namespace='/')
    else:
        print(f"⚠️ Call {call_id} not found in active calls")

# 2. زيد هاد الدالة الجديدة تحتها مباشرة:
@socketio.on('admin_join_call')
def handle_admin_join_call(data):
    """
    ✅ NEW: Admin وصل لصفحة الفيديو، نعطيه الـ Offer اللي مخبي ليه
    """
    call_id = data.get('call_id')
    print(f"👨‍⚕️ Admin joining call page: {call_id} with new SID: {request.sid}")
    
    if call_id in active_calls:
        # تحديث الـ SID الجديد للـ Admin
        active_calls[call_id]['admin_sid'] = request.sid
        
        # الانضمام للغرف
        join_room(call_id)
        
        # ✅ واش كاين شي Offer مخبي؟ صيفطو دابا!
        cached_offer = active_calls[call_id].get('offer')
        user_id = active_calls[call_id].get('user_id')
        
        if cached_offer:
            print(f"📦 Found CACHED Offer. Sending to Admin now...")
            emit('incoming_webrtc_offer', {
                'call_id': call_id,
                'user_id': user_id,
                'offer': cached_offer
            }, room=request.sid, namespace='/')
        else:
            print("⏳ No cached offer yet. Waiting for Flutter...")






@socketio.on('call_rejected')
def handle_call_rejected(data):
    """
    Patient رفض المكالمة
    """
    call_id = data.get('call_id')
    
    if call_id in active_calls:
        admin_sid = active_calls[call_id].get('admin_sid')
        
        # إعلام الـ admin
        if admin_sid:
            emit('call_rejected', {
                'call_id': call_id
            }, room=admin_sid, namespace='/')
        
        # حذف المكالمة
        del active_calls[call_id]
        
        print(f"❌ Call {call_id} rejected")


@socketio.on('webrtc_offer')
def handle_webrtc_offer(data):
    """
    ✅ FIXED: تخزين Offer إذا كان Admin غير متصل (Caching)
    """
    call_id = data.get('call_id')
    user_id = data.get('user_id')
    offer = data.get('offer')
    
    print(f"📩 Received offer for call {call_id} from user {user_id}")
    
    if call_id in active_calls:
        # 1. تخزين الـ Offer في الذاكرة (لأن Admin قد يكون في مرحلة تحميل الصفحة)
        active_calls[call_id]['offer'] = offer
        print(f"💾 Offer CACHED for call {call_id} (Waiting for Admin to rejoin)")

        admin_sid = active_calls[call_id].get('admin_sid')
        
        # 2. محاولة إرساله إذا كان Admin موجوداً حالياً
        emit('incoming_webrtc_offer', {
            'call_id': call_id,
            'user_id': user_id,
            'offer': offer
        }, room=call_id, namespace='/')
        
        if admin_sid:
            emit('incoming_webrtc_offer', {
                'call_id': call_id,
                'user_id': user_id,
                'offer': offer
            }, room=admin_sid, namespace='/')
        
        print(f"📤 Offer broadcasted to room {call_id}")
    else:
        print(f"⚠️ Call {call_id} not found in active calls")


@socketio.on('webrtc_answer')
def handle_webrtc_answer(data):
    """
    ✅ FIXED: استقبال ANSWER من admin وإرساله للـ Flutter
    """
    user_id = data.get('user_id')
    call_id = data.get('call_id')  # ✅ Admin should send call_id
    answer = data.get('answer')
    
    print(f"📩 Received answer from admin for user {user_id}")
    
    # ✅ Send to user's room
    emit('webrtc_answer', {
        'from_user_id': 'admin',
        'answer': answer
    }, room=user_id, namespace='/')
    
    print(f"📤 Answer sent to user {user_id}")


@socketio.on('ice_candidate')
def handle_ice_candidate(data):
    """
    ✅ FIXED: تبادل ICE candidates - target يمكن يكون call_id أو user_id
    """
    target = data.get('target')
    candidate = data.get('candidate')
    
    print(f"🧊 ICE candidate for target: {target}")
    
    # ✅ Send to target room (can be call_id or user_id)
    emit('ice_candidate', {
        'candidate': candidate
    }, room=target, namespace='/')
    
    print(f"✅ ICE candidate forwarded to room {target}")




@socketio.on('connect')
def handle_connect():
    """
    عند الاتصال بالـ socket
    """
    print(f"✅ Client connected: {request.sid}")


@socketio.on('disconnect')
def handle_disconnect():
    """
    عند قطع الاتصال
    """
    print(f"❌ Client disconnected: {request.sid}")
    
    # تنظيف المكالمات النشطة
    calls_to_remove = []
    for call_id, call_data in active_calls.items():
        if call_data.get('admin_sid') == request.sid:
            # إعلام الـ patient
            patient_id = call_data.get('user_id')
            if patient_id:
                emit('call_ended', {
                    'call_id': call_id,
                    'reason': 'admin_disconnected'
                }, room=patient_id, namespace='/')
            
            calls_to_remove.append(call_id)
    
    # حذف المكالمات
    for call_id in calls_to_remove:
        del active_calls[call_id]
        print(f"🗑️ Removed call {call_id} due to admin disconnect")


@socketio.on('register_user')
def handle_register_user(data):
    """
    ✅ FIXED: تسجيل user_id مع socket session + join room
    """
    user_id = data.get('user_id')
    
    # انضمام للـ room بناءً على user_id
    join_room(user_id)
    
    print(f"✅ User {user_id} registered with SID {request.sid}")
    
    emit('registration_success', {
        'user_id': user_id,
        'sid': request.sid
    })

    @socketio.on('join_room')
    def handle_join_room(data):
        """
        Allow clients to manually join a room
        """
        room = data.get('room')
        if room:
            join_room(room)
            print(f"🚪 Client {request.sid} joined room: {room}")
            emit('room_joined', {'room': room})















@socketio.on('connect')
def handle_connect():
    print("🔌 A user connected via Socket.IO")


@socketio.on('disconnect')
def handle_disconnect():
    print("🔌 A user disconnected from Socket.IO")



# ============================================================
# 📄 Admin HTML Pages (Dashboard & User Sessions)
# ============================================================

@app.route('/admin/dashboard', methods=['GET'])
def admin_dashboard():
    return send_from_directory('static', 'admin_dashboard.html')


@app.route('/admin/user_sessions')
def admin_user_sessions():
    return app.send_static_file('user_sessions.html')


# ============================================================
# ✅ Save Pose (full pipeline)
# ============================================================

@app.route('/save-pose-complete', methods=['POST'])
def save_pose_complete():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        image_base64 = data.get('image_base64')
        analysis_data = data.get('analysis', {})
        poses_data = data.get('poses', [])
        record_id = data.get('record_id', str(uuid.uuid4()))

        image_size_data = data.get('image_size')

        if not user_id:
            return jsonify({"error": "user_id is required"}), 400

        image_path = None
        if image_base64:
            timestamp_file = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"pose_{user_id[:8]}_{timestamp_file}.jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            try:
                if ',' in image_base64:
                    image_base64 = image_base64.split(',')[1]
                image_data = base64.b64decode(image_base64)
                with open(filepath, 'wb') as f:
                    f.write(image_data)
                image_path = f"/uploads/{filename}"
                print(f"✅ Image saved to: {filepath}")
            except Exception as e:
                print(f"⚠️ Could not save image for record {record_id}: {e}")

        timestamp_str = data.get('timestamp', datetime.datetime.now().isoformat())
        shoulder_y_diff = analysis_data.get('shoulder_y_diff')
        hip_y_diff = analysis_data.get('hip_y_diff')
        shoulder_higher_side = analysis_data.get('shoulder_higher_side')
        hip_higher_side = analysis_data.get('hip_higher_side')

        # Add image dimensions into analysis_json
        if (image_size_data and isinstance(image_size_data, dict)
                and 'width' in image_size_data and 'height' in image_size_data):
            analysis_data['image_dimensions'] = image_size_data
            print(f"ℹ️ Added image dimensions to analysis data: {image_size_data}")
        else:
            print(f"⚠️ Image size data not found or invalid in payload for {record_id}.")

        poses_json_str = json.dumps(poses_data)
        analysis_json_str = json.dumps(analysis_data)

        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "DB connection failed"}), 503

        cursor = conn.cursor()
        try:
            query = """
                INSERT INTO poses (
                    id, user_id, timestamp, image_path,
                    shoulder_y_diff, hip_y_diff,
                    shoulder_higher_side, hip_higher_side,
                    poses_json, analysis_json, admin_comment
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (
                record_id, user_id,
                datetime.datetime.fromisoformat(timestamp_str),
                image_path,
                shoulder_y_diff, hip_y_diff,
                shoulder_higher_side, hip_higher_side,
                poses_json_str, analysis_json_str, ""
            )
            cursor.execute(query, values)
            conn.commit()
            cursor.close()

            print(f"✅ Pose data saved for record ID: {record_id}")
            return jsonify({
                "success": True,
                "message": "Pose saved successfully",
                "record_id": record_id,
                "image_path": image_path
            }), 200
        except Exception as db_err:
            cursor.close()
            print(f"❌ DB Error saving pose {record_id}: {db_err}")
            return jsonify({"error": f"Database error: {db_err}"}), 500

    except Exception as e:
        print(f"❌ Error in /save-pose-complete: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ============================================================
# 🤖 AI/NLP Endpoints
# ============================================================

@app.route('/generate-report', methods=['POST'])
def generate_report():
    try:
        data = request.get_json()
    except Exception as json_err:
        print(f"❌ Invalid JSON in /generate-report: {json_err}")
        return jsonify({"error": "Invalid JSON payload"}), 400

    if not data or 'analysis' not in data:
        return jsonify({"error": "Missing analysis data"}), 400

    analysis_data = data['analysis']

    try:
        report = posture_analyzer.generate_report(analysis_data)
        summary = posture_analyzer.get_quick_summary(analysis_data)
        print(f"✅ Report generated - Severity: {summary.get('severity_label', 'N/A')}")

        return jsonify({
            "success": True,
            "report": report,
            "summary": summary,
            "timestamp": datetime.datetime.now().isoformat()
        }), 200

    except Exception as e:
        print(f"❌ Error generating report: {e}")
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to generate report: {str(e)}"
        }), 500


@app.route('/quick-analysis', methods=['POST'])
def quick_analysis():
    try:
        data = request.get_json()
    except Exception as json_err:
        print(f"❌ Invalid JSON in /quick-analysis: {json_err}")
        return jsonify({"error": "Invalid JSON payload"}), 400

    if not data or 'analysis' not in data:
        return jsonify({"error": "Missing analysis data"}), 400

    try:
        summary = posture_analyzer.get_quick_summary(data['analysis'])
        return jsonify(summary), 200
    except Exception as e:
        print(f"❌ Quick analysis error: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================
# 3D Processing
# ============================================================

@app.route('/process-3d-pose', methods=['POST'])
def process_3d_pose():
    results = None

    try:
        data = request.get_json()
    except Exception as json_err:
        print(f"❌ Invalid JSON in /process-3d-pose: {json_err}")
        return jsonify({"error": "Invalid JSON payload"}), 400

    if not data or 'image_base64' not in data:
        return jsonify({"error": "Missing 'image_base64' data in request"}), 400

    try:
        image_base64_data = data['image_base64']
        if ',' in image_base64_data:
            image_base64_data = image_base64_data.split(',')[1]
        image_data = base64.b64decode(image_base64_data)

        np_arr = np.frombuffer(image_data, np.uint8)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image_bgr is None:
            print("❌ Could not decode image in /process-3d-pose")
            return jsonify({"error": "Could not decode image"}), 400

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose_processor:
            results = pose_processor.process(image_rgb)

    except Exception as e:
        print(f"❌ Image processing failed in /process-3d-pose: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Image processing failed: {e}"}), 500

    if not results or not results.pose_world_landmarks:
        print("ℹ️ No pose detected in the image for 3D processing.")
        return jsonify({"error": "No pose detected in the image"}), 404

    landmarks_3d = []
    for i, landmark in enumerate(results.pose_world_landmarks.landmark):
        landmarks_3d.append({
            "id": i,
            "name": mp_pose.PoseLandmark(i).name,
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z,
            "visibility": landmark.visibility
        })

    print(f"✅ Successfully processed 3D pose, found {len(landmarks_3d)} landmarks.")
    return jsonify({"landmarks": landmarks_3d}), 200


# ============================================================
# Static Files (viewer + uploads)
# ============================================================

@app.route('/viewer')
def serve_viewer():
    return send_from_directory('static', 'viewer.html')


@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('static/js', filename)


@app.route('/uploads/<filename>')
def serve_upload(filename):
    try:
        return send_from_directory(UPLOAD_FOLDER, filename)
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        print(f"❌ Error serving file {filename}: {e}")
        return jsonify({"error": "Internal server error"}), 500


# ============================================================
# Authentication & Admin
# ============================================================

@app.route('/admin/get-users', methods=['GET'])
def admin_get_users():
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "DB connection failed"}), 503

    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id, name, email FROM users WHERE role = 'user' OR role IS NULL")
        users = cursor.fetchall()
        cursor.close()

        return jsonify({
            "success": True,
            "total_users": len(users),
            "users": users
        }), 200
    except Exception as e:
        cursor.close()
        print(f"❌ DB Error in /admin/get-users: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    if not data or not all(k in data for k in ['name', 'email', 'password', 'birth_date']):
        return jsonify({"error": "Missing required fields"}), 400

    password_hash = generate_password_hash(data['password'])
    user_id = str(uuid.uuid4())

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "DB connection failed"}), 503

    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (id, name, email, password_hash, birth_date) VALUES (%s, %s, %s, %s, %s)",
            (user_id, data['name'], data['email'], password_hash, data['birth_date'])
        )
        conn.commit()
        cursor.close()
        print(f"✅ User created: {data['email']}")
        return jsonify({"message": "User created", "user_id": user_id}), 201
    except mysql.connector.IntegrityError:
        cursor.close()
        print(f"⚠️ Signup failed: Email {data['email']} already exists.")
        return jsonify({"error": "Email already exists"}), 409
    except Exception as e:
        cursor.close()
        print(f"❌ DB Error during signup: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or not all(k in data for k in ['email', 'password']):
        return jsonify({"error": "Missing email or password"}), 400

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "DB connection failed"}), 503

    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM users WHERE email = %s", (data['email'],))
        user = cursor.fetchone()
        cursor.close()

        if user and check_password_hash(user['password_hash'], data['password']):
            if isinstance(user.get('birth_date'), datetime.date):
                user['birth_date'] = user['birth_date'].isoformat()
            print(f"✅ User logged in: {data['email']}")
            return jsonify({"message": "Login successful", "user": user}), 200
        else:
            print(f"⚠️ Login failed for: {data['email']}")
            return jsonify({"error": "Invalid email or password"}), 401
    except Exception as e:
        cursor.close()
        print(f"❌ DB Error during login: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/admin/get-all-poses', methods=['GET'])
def admin_get_all_poses():
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "DB connection failed"}), 503

    cursor = conn.cursor(dictionary=True)
    try:
        query = """
            SELECT
                p.id, p.user_id, u.name as user_name, u.email as user_email,
                p.timestamp, p.image_path,
                p.shoulder_y_diff, p.hip_y_diff,
                p.shoulder_higher_side, p.hip_higher_side,
                p.poses_json, p.analysis_json, p.admin_comment
            FROM poses p
            INNER JOIN users u ON p.user_id = u.id
            ORDER BY p.timestamp DESC
        """
        cursor.execute(query)
        poses = cursor.fetchall()
        cursor.close()

        results = []
        for row in poses:
            reconstructed = {
                "id": row.get('id'),
                "user_id": row.get('user_id'),
                "user_name": row.get('user_name'),
                "user_email": row.get('user_email'),
                "timestamp": row['timestamp'].isoformat() if row.get('timestamp') else None,
                "image_path": row.get('image_path'),
                "shoulder_y_diff": float(row['shoulder_y_diff']) if row.get('shoulder_y_diff') else None,
                "hip_y_diff": float(row['hip_y_diff']) if row.get('hip_y_diff') else None,
                "shoulder_higher_side": row.get('shoulder_higher_side'),
                "hip_higher_side": row.get('hip_higher_side'),
                "poses": json.loads(row['poses_json']) if row.get('poses_json') and isinstance(row['poses_json'], str) else [],
                "analysis": json.loads(row['analysis_json']) if row.get('analysis_json') and isinstance(row['analysis_json'], str) else {},
                "admin_comment": row.get('admin_comment')
            }
            results.append(reconstructed)

        return jsonify({
            "success": True,
            "total_poses": len(results),
            "poses": results
        }), 200
    except Exception as e:
        cursor.close()
        print(f"❌ DB Error in /admin/get-all-poses: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/admin/stats', methods=['GET'])
def admin_stats():
    """
    ✅ FIXED: Returns total users, total poses, and TODAY'S UPLOADS
    """
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "DB connection failed"}), 503

    cursor = conn.cursor(dictionary=True)
    try:
        # 1️⃣ Total users (excluding admin)
        cursor.execute("SELECT COUNT(*) as total FROM users WHERE role='user' OR role IS NULL")
        users_count = cursor.fetchone()['total']

        # 2️⃣ Total poses
        cursor.execute("SELECT COUNT(*) as total FROM poses")
        poses_count = cursor.fetchone()['total']

        # 3️⃣ ✅ TODAY'S UPLOADS (THIS WAS MISSING!)
        # Get today's date in UTC (start of day)
        today = datetime.datetime.utcnow().date()
        cursor.execute("""
            SELECT COUNT(*) as total FROM poses 
            WHERE DATE(timestamp) = %s
        """, (today,))
        today_uploads = cursor.fetchone()['total']

        # 4️⃣ Recent uploads (for reference)
        cursor.execute("""
            SELECT p.timestamp, u.name
            FROM poses p
            JOIN users u ON p.user_id = u.id
            ORDER BY p.timestamp DESC
            LIMIT 10
        """)
        recent = cursor.fetchall()

        # Convert timestamps to ISO format for JSON
        for r in recent:
            if r.get('timestamp') and isinstance(r['timestamp'], datetime.datetime):
                r['timestamp'] = r['timestamp'].isoformat()

        cursor.close()

        # ✅ Return all stats including today_uploads
        return jsonify({
            "total_users": users_count,
            "total_poses": poses_count,
            "today_uploads": today_uploads,
            "recent_uploads": recent
        }), 200

    except Exception as e:
        cursor.close()
        print(f"❌ DB Error in /admin/stats: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ============================================================
# Pose Data Management
# ============================================================

@app.route('/get-poses/<user_id>', methods=['GET'])
def get_poses(user_id):
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "DB connection failed"}), 503

    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT * FROM poses WHERE user_id = %s ORDER BY timestamp DESC",
            (user_id,)
        )
        poses_from_db = cursor.fetchall()
        cursor.close()

        results = []
        for row in poses_from_db:
            reconstructed_data = {
                "id": row.get('id'),
                "user_id": row.get('user_id'),
                "timestamp": row['timestamp'].isoformat() if row.get('timestamp') else None,
                "image_path": row.get('image_path'),
                "poses": json.loads(row['poses_json']) if row.get('poses_json') and isinstance(row['poses_json'], str) else [],
                "analysis": json.loads(row['analysis_json']) if row.get('analysis_json') and isinstance(row['analysis_json'], str) else {},
                "admin_comment": row.get('admin_comment')
            }
            results.append(reconstructed_data)
        print(f"✅ Fetched {len(results)} poses for user {user_id}")
        return jsonify(results), 200
    except Exception as e:
        cursor.close()
        print(f"❌ DB Error in /get-poses/{user_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/update-comment', methods=['POST'])
def update_comment():
    data = request.get_json()
    record_id = data.get('record_id')
    comment_text = data.get('comment')
    if not record_id or comment_text is None:
        return jsonify({"error": "Missing record_id or comment"}), 400

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "DB connection failed"}), 503

    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE poses SET admin_comment = %s WHERE id = %s",
            (comment_text, record_id)
        )
        conn.commit()
        rows_affected = cursor.rowcount
        cursor.close()
        if rows_affected > 0:
            print(f"✅ Comment updated for record ID: {record_id}")
            return jsonify({"message": "Comment updated successfully"}), 200
        else:
            print(f"⚠️ No record found with ID: {record_id} to update comment.")
            return jsonify({"error": "Record not found"}), 404
    except Exception as e:
        cursor.close()
        print(f"❌ DB Error updating comment for {record_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/delete-pose/<pose_id>', methods=['DELETE'])
def delete_pose(pose_id):
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 503

    cursor = conn.cursor()
    try:
        cursor.execute("SELECT image_path FROM poses WHERE id = %s", (pose_id,))
        result = cursor.fetchone()
        if result and result[0]:
            image_file_path = os.path.join(UPLOAD_FOLDER, os.path.basename(result[0]))
            if os.path.exists(image_file_path):
                try:
                    os.remove(image_file_path)
                    print(f"🗑️ Deleted image file: {image_file_path}")
                except Exception as file_err:
                    print(f"⚠️ Could not delete image file {image_file_path}: {file_err}")

        query = "DELETE FROM poses WHERE id = %s"
        cursor.execute(query, (pose_id,))
        conn.commit()
        rows_affected = cursor.rowcount
        cursor.close()

        if rows_affected > 0:
            print(f"✅ Deleted pose with ID: {pose_id}")
            return jsonify({"message": "Pose deleted successfully"}), 200
        else:
            print(f"⚠️ No record found with ID: {pose_id} to delete.")
            return jsonify({"error": "Record not found"}), 404
    except Exception as e:
        cursor.close()
        print(f"❌ DB Error deleting pose {pose_id}: {e}")
        return jsonify({"error": "Database error", "details": str(e)}), 500


# ============================================================
# 🤖 Chatbot (Hugging Face + fallback rule-based)
# ============================================================

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if not HUGGINGFACE_API_KEY:
    print("⚠️ HUGGINGFACE_API_KEY not found in .env. Chatbot might fall back to rule-based.")

HUGGINGFACE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
HUGGINGFACE_API_URL = f"https://api-inference.huggingface.co/models/{os.getenv('HUGGINGFACE_MODEL', HUGGINGFACE_MODEL)}"


@app.route('/chatbot-response', methods=['POST'])
def chatbot_response():
    try:
        data = request.get_json()
    except Exception as json_err:
        print(f"❌ Invalid JSON in /chatbot-response: {json_err}")
        return jsonify({"error": "Invalid JSON data"}), 400

    user_message = data.get('user_message', '')
    analysis = data.get('analysis', {})

    if not user_message:
        return jsonify({"error": "User message missing"}), 400

    try:
        context = _build_chatbot_context(analysis)
        response_text = _get_chatbot_response(user_message, context)

        print(f"✅ Chatbot response generated for: '{user_message[:30]}...'")
        return jsonify({
            "success": True,
            "response": response_text,
            "timestamp": datetime.datetime.now().isoformat()
        }), 200

    except Exception as e:
        print(f"❌ Chatbot error: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "response": "Sorry, an error occurred while processing your message. Please try again.",
            "error": str(e)
        }), 500


def _build_chatbot_context(analysis):
    shoulder = analysis.get('shoulder_y_diff', 0) or 0
    hip = analysis.get('hip_y_diff', 0) or 0

    context = f"""You are an experienced physiotherapist providing advice based on posture analysis data.

    Patient Information:
    - Shoulder vertical difference: {shoulder:.1f} mm
    - Hip vertical difference: {hip:.1f} mm
    - Higher shoulder side: {analysis.get('shoulder_higher_side', 'N/A')}
    - Higher hip side: {analysis.get('hip_higher_side', 'N/A')}

    Instructions for your response:
    1. Language: Use clear and simple English, but incorporate relevant physiotherapy terms where appropriate.
    2. Role: Maintain the perspective of a knowledgeable and caring physiotherapist.
    3. Recommendations: Provide specific, actionable exercise suggestions tailored to the analysis.
    4. Explanations: Briefly explain the potential implications of the detected imbalances.
    5. Safety: Include warnings about stopping if pain occurs and starting gently.
    6. Consultation: Advise consulting a doctor or physical therapist for persistent issues or severe pain.
    7. Length: Keep the response concise, ideally under 200 words.
    8. Formatting: Use bullet points for exercises or tips.

    ---
    Patient's question:"""
    return context


def _get_chatbot_response(user_message, context):
    if not HUGGINGFACE_API_KEY or HUGGINGFACE_API_KEY == "hf_default_key":
        print("ℹ️ No Hugging Face API key. Using rule-based chatbot.")
        return _get_rule_based_response(user_message)

    try:
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": f"{context}\n{user_message}",
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False
            }
        }

        response = requests.post(
            HUGGINGFACE_API_URL,
            headers=headers,
            json=payload,
            timeout=20
        )
        response.raise_for_status()

        result = response.json()
        if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
            answer = result[0]['generated_text'].strip()
            if answer.startswith("Patient's question:"):
                answer = answer.split("Patient's question:")[-1].strip()
            if answer:
                print("✅ Got response from Hugging Face API.")
                return answer
            else:
                print("⚠️ Hugging Face API returned empty response.")
        else:
            print(f"⚠️ Unexpected response format from Hugging Face API: {result}")

    except requests.Timeout:
        print("⚠️ Hugging Face API timeout, switching to local model")
    except requests.RequestException as req_err:
        print(f"⚠️ Hugging Face API request error: {req_err}")
    except Exception as e:
        print(f"⚠️ Hugging Face API unexpected error: {e}")
        traceback.print_exc()

    print("ℹ️ Falling back to rule-based chatbot.")
    return _get_rule_based_response(user_message)


def _get_rule_based_response(user_message):
    msg_lower = user_message.lower()

    if any(word in msg_lower for word in ['neck pain', 'neck', 'cervical', 'stiff neck']):
        return """Neck pain often relates to posture, especially forward head posture or imbalances.

- Chin tucks: 10 reps, hold 5s.
- Neck side bends: 15–30s each side.
- Upper trapezius stretches.

Stop if you feel sharp pain and consult a doctor if symptoms persist."""

    if any(word in msg_lower for word in ['back pain', 'lower back', 'spine', 'backache', 'lumbar']):
        return """Lower back pain is often linked to weak core or hip imbalance.

- Pelvic tilts.
- Bridges.
- Cat–cow stretch.

Keep movements slow, avoid heavy lifting, and seek medical help if pain radiates to legs or is severe."""

    if any(word in msg_lower for word in ['exercise', 'workout', 'training', 'stretch', 'strengthen']):
        return """To support posture:

Stretches:
- Chest doorway stretch.
- Hip flexor stretch.
- Hamstring stretch.

Strength:
- Rows.
- Glute bridges.
- Planks.

Start gently and be consistent 2–3x/week."""

    if any(word in msg_lower for word in ['posture', 'slouch', 'shoulder', 'alignment', 'stand straight']):
        return """For better posture:

- Keep ears over shoulders, shoulders relaxed back and down.
- Adjust desk/monitor to eye level.
- Strengthen core and upper back.
- Stretch chest and front shoulders.

Think: tall spine, relaxed breathing, frequent posture checks."""

    if any(word in msg_lower for word in ['doctor', 'emergency', 'examination', 'consult', 'specialist', 'pain severe']):
        return """Consult a doctor or physiotherapist if:

- Pain is severe or from trauma.
- There is numbness, tingling, or weakness.
- Pain lasts >4–6 weeks despite self-care.

They can provide a full assessment and tailored plan."""

    return """Hello 👋 I'm your physiotherapy assistant.

You can ask me about:
- Exercises for neck or back.
- How to improve posture.
- When to see a doctor.

Tell me what bothers you most and I'll guide you."""


# ============================================================
# Testing Endpoints
# ============================================================

@app.route('/test', methods=['GET'])
def test_connection():
    return jsonify({
        "message": "Server is working!",
        "status": "success",
        "ai_system": "active",
        "database_status": "Connected" if db_connection and db_connection.is_connected() else "Disconnected"
    })


@app.route('/test-ai', methods=['GET', 'POST'])
def test_ai():
    try:
        test_data = {
            'shoulder_y_diff': 7.5,
            'hip_y_diff': 4.2,
            'shoulder_higher_side': 'Left Shoulder Higher',
            'hip_higher_side': 'Left Hip Higher'
        }
        summary = posture_analyzer.get_quick_summary(test_data)
        report = posture_analyzer.generate_report(test_data)

        return jsonify({
            "status": "AI system working correctly",
            "test_summary": summary,
            "test_report_sample": report[:200] + "..."
        }), 200
    except Exception as e:
        print(f"❌ Error during AI test: {e}")
        traceback.print_exc()
        return jsonify({
            "status": "AI system error",
            "error": str(e)
        }), 500


@app.route('/test-chatbot', methods=['GET', 'POST'])
def test_chatbot():
    if request.method == 'GET':
        return jsonify({
            "status": "Chatbot active",
            "model_preference": "Hugging Face API (fallback to Rule-Based)",
            "message": "Use POST with JSON body to test interaction.",
            "example_payload": {
                "user_message": "What should I do to correct my posture?",
                "analysis": {
                    "shoulder_y_diff": 7.5, "hip_y_diff": 4.2,
                    "shoulder_higher_side": "Left", "hip_higher_side": "Left"
                }
            }
        }), 200

    try:
        data = request.get_json()
        user_msg = data.get('user_message', 'Tell me about posture.')
        analysis = data.get('analysis', {'shoulder_y_diff': 5.0, 'hip_y_diff': 3.0})

        context = _build_chatbot_context(analysis)
        response = _get_chatbot_response(user_msg, context)

        return jsonify({
            "status": "success",
            "user_message": user_msg,
            "analysis_context": analysis,
            "chatbot_response": response
        }), 200
    except Exception as e:
        print(f"❌ Error during Chatbot test: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ============================================================
# Main Execution
# ============================================================

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🚀 Starting Pose Analysis Server")
    print("=" * 50)
    
    host_ip = '0.0.0.0'
    
    # ✅✅ التغيير المهم: قراءة PORT من Render
    # Render كيعطي بورت عشوائي، خاصنا نخدمو عليه
    port = int(os.environ.get("PORT", 5000))
    
    print(f"🌍 Server listening on http://{host_ip}:{port}")

    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"   (Accessible on your network at http://{local_ip}:{port})")
    except Exception:
        print("   (Could not determine local network IP)")

    # طباعة البورت باش نتأكدو (للتجربة)
    print(f"📦 Database Port Configured: {os.getenv('MYSQL_PORT', 'Not Set (Default 3306)')}")
    print("=" * 50 + "\n")

    # Run with Socket.IO
    socketio.run(
        app,
        host=host_ip,
        port=port,  # ⬅️ استعمال المتغير port
        debug=False,
        allow_unsafe_werkzeug=True
    )
