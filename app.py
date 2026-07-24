from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import os
import base64
import cv2
import numpy as np

from sms_service import send_sms

from telegram_service import (
    get_telegram_start_connections,
    get_telegram_start_link,
    send_telegram_alert,
)
from datetime import datetime

from drowsiness import run_drowsiness_frame

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

PROCESSED_FOLDER = "processed"
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# ===========================
# STORE LAST RESULTS
# ===========================

last_overtaking_result = {
    "score": 0,
    "safe": 0,
    "rash": 0
}

last_drowsiness_status = "NORMAL"

# ===========================
# OVERTAKING ROUTE
# ===========================

@app.route("/overtaking", methods=["POST"])
def overtaking():

    global last_overtaking_result

    if "file" not in request.files:

        print("No file found in request")
 
        return jsonify({
            "error": "No file received"
        }), 400

    file = request.files["file"]

    if file.filename == "":

        print("Empty filename")

        return jsonify({
            "error": "Empty file"
        }), 400

    video_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(video_path)

    print("Video received:", video_path)

    # Import model
    from inference import run_overtaking

    result = run_overtaking(video_path)

    if "video_filename" in result:
        result["video_url"] = (
            request.host_url.rstrip("/")
            + "/processed/"
            + result["video_filename"]
        )

    print("Overtaking Result:", result)

    # SAVE RESULT
    last_overtaking_result["score"] = result["score"]
    last_overtaking_result["safe"] = result["safe"]
    last_overtaking_result["rash"] = result["rash"]

    return jsonify(result)

@app.route("/processed/<filename>", methods=["GET"])
def processed_video(filename):

    return send_from_directory(
        PROCESSED_FOLDER,
        filename,
        mimetype="video/mp4"
    )

# ===========================
# DROWSINESS ROUTE
# ===========================

@app.route("/drowsiness_frame", methods=["POST"])
def drowsiness_frame():

    global last_drowsiness_status

    data = request.json

    image_data = data["image"]

    # Decode image
    img_bytes = base64.b64decode(image_data)

    np_arr = np.frombuffer(
        img_bytes,
        np.uint8
    )

    frame = cv2.imdecode(
        np_arr,
        cv2.IMREAD_COLOR
    )

    result = run_drowsiness_frame(frame)

    print("Drowsiness:", result)

    # SAVE STATUS
    last_drowsiness_status = result["alert"]

    return jsonify(result)

# ===========================
# TELEGRAM ALERTS
# ===========================

@app.route("/telegram/link", methods=["GET"])
def telegram_link():

    driver_id = request.args.get("driverId")

    if not driver_id:
        return jsonify({
        "success": False,
        "message": "driverId is required"
    }), 400
    contact_id = request.args.get("contactId")

    if not contact_id:
        return jsonify({
            "success": False,
            "error": "contactId is required."
        }), 400

    success, result = get_telegram_start_link(driver_id, contact_id)

    if not success:
        return jsonify({
            "success": False,
            "error": result
        }), 500

    return jsonify({
        "success": True,
        "link": result
    })


@app.route("/telegram/connections", methods=["GET"])
def telegram_connections():

    driver_id = request.args.get("driverId")

    if not driver_id:
      return jsonify({
        "success": False,
        "message": "driverId is required"
    }), 400
    success, result = get_telegram_start_connections(driver_id)

    if not success:
        return jsonify({
            "success": False,
            "error": result
        }), 502

    return jsonify({
        "success": True,
        "connections": result
    })


@app.route("/emergency", methods=["POST"])
def emergency():

    print("========== EMERGENCY API HIT ==========")

    data = request.json

    print("========== EMERGENCY DATA ==========")
    print(data)
    print("Latitude:", data.get("latitude"))
    print("Longitude:", data.get("longitude"))
    print("====================================")

    driver = data.get("driver", "Unknown Driver")
    driver_id = data.get("driverId")

    if not driver_id:
        return jsonify({
        "success": False,
        "message": "driverId is required"
    }), 400

    latitude = data.get("latitude")
    longitude = data.get("longitude")

    time_now = datetime.now().strftime("%d-%b-%Y %I:%M %p")

    if latitude and longitude:
        location = f"https://maps.google.com/?q={latitude},{longitude}"
    else:
        location = "Location unavailable"

    if latitude and longitude:
        live_tracking = f"{request.host_url.rstrip('/')}/live/{driver_id}"
    else:
        live_tracking = "Live tracking unavailable"    


    message = f"""
🚨 DRIVER SAFETY ALERT 🚨

👤 Driver:
{driver}

⚠️ Status:
Driver did not respond to the drowsiness alarm.

🕒 Time:
{time_now}

📍 Current Location:
{location}

🌍 Live Tracking:
{live_tracking}

Please contact the driver immediately.
"""

    contacts = (
        db.collection("Drivers")
        .document(driver_id)
        .collection("emergencyContacts")
        .stream()
    )

    results = []

    for contact in contacts:

        contact_data = contact.to_dict()

        print(contact_data)

        result = {
            "name": contact_data.get("name"),
            "telegram": False,
            "sms": False,
        }

        # Telegram
        if (
            contact_data.get("telegramConnected")
            and contact_data.get("telegramChatId")
        ):

            success, response = send_telegram_alert(
                contact_data["telegramChatId"],
                message,
            )

            print(
                f"Telegram -> {contact_data.get('name')} : {response}"
            )

            result["telegram"] = success

        # SMS
        if contact_data.get("phone"):

            sms_success, sms_response = send_sms(
                contact_data["phone"],
                message,
            )

            print(
                f"SMS -> {contact_data.get('name')} : {sms_response}"
            )

            result["sms"] = sms_success

        results.append(result)

    return jsonify({
        "success": True,
        "notifications": results
    })

# ===========================
# UPDATE LOCATION ROUTE
# ===========================


@app.route("/update-location", methods=["POST"])
def update_location():
    try:
        data = request.json

        driver_id = data.get("driverId")
        latitude = data.get("latitude")
        longitude = data.get("longitude")

        if not driver_id or latitude is None or longitude is None:
            return jsonify({
                "success": False,
                "message": "Missing required fields"
            }), 400

        print(f"Updating location for {driver_id}: {latitude}, {longitude}")

        db.collection("Drivers").document(driver_id).set({
            "liveLocation": {
                "latitude": latitude,
                "longitude": longitude,
                "timestamp": datetime.utcnow(),
                "emergency": True
            }
        }, merge=True)

        return jsonify({
            "success": True,
            "message": "Location updated successfully"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


# ===========================
# GET LIVE LOCATION
# ===========================

@app.route("/live-location/<driver_id>", methods=["GET"])
def get_live_location(driver_id):

    try:

        doc = db.collection("Drivers").document(driver_id).get()

        if not doc.exists:
            return jsonify({
                "success": False,
                "message": "Driver not found"
            }), 404

        data = doc.to_dict()

        location = data.get("liveLocation")

        if not location:
            return jsonify({
                "success": False,
                "message": "No live location available"
            }), 404

        return jsonify({
            "success": True,
            "latitude": location.get("latitude"),
            "longitude": location.get("longitude"),
            "timestamp": location.get("timestamp"),
            "emergency": location.get("emergency"),
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

# ===========================
# LIVE TRACKING PAGE
# ===========================

@app.route("/live/<driver_id>")
def live_tracking(driver_id):

    return render_template(
        "live_tracking.html",
        driver_id=driver_id
    )

# ===========================
# DASHBOARD ROUTE
# ===========================

@app.route("/dashboard", methods=["GET"])
def dashboard():

    return jsonify({

        "score":
            last_overtaking_result["score"],

        "safe":
            last_overtaking_result["safe"],

        "rash":
            last_overtaking_result["rash"],

        "status":
            last_drowsiness_status

    })


@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "success": True,
        "status": "ok"
    })

# ===========================
# RUN SERVER
# ===========================

if __name__ == "__main__":

    print("Starting Flask Server...")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
