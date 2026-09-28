from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from werkzeug.utils import secure_filename

import sqlite3
import os
from datetime import datetime


# =========================================================
# AGRIVISION AI - FLASK BACKEND
# =========================================================

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)


# =========================================================
# PROJECT PATHS
# =========================================================

# This automatically points to the current backend folder.
# Local:
# C:\...\Desktop\AgriVision-Backend
#
# Render:
# /opt/render/project/src
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# YOLO model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "best.pt"
)


# Uploaded images
UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)


# YOLO result images
RESULT_FOLDER = os.path.join(
    BASE_DIR,
    "results"
)


# Detection result images
DETECTION_FOLDER = os.path.join(
    RESULT_FOLDER,
    "detections"
)


# SQLite database
DATABASE = os.path.join(
    BASE_DIR,
    "agrivision.db"
)


# =========================================================
# CREATE REQUIRED FOLDERS
# =========================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)

os.makedirs(
    DETECTION_FOLDER,
    exist_ok=True
)


# =========================================================
# ALLOWED IMAGE EXTENSIONS
# =========================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}


def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# =========================================================
# LOAD YOLO MODEL
# =========================================================

print("Loading AgriVision AI model...")

try:

    model = YOLO(MODEL_PATH)

    print("YOLO model loaded successfully!")

except Exception as error:

    print(
        "ERROR: Could not load YOLO model."
    )

    print(
        error
    )

    raise


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

def init_database():

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS detection_history (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            image_name TEXT NOT NULL,

            crop_count INTEGER NOT NULL,

            weed_count INTEGER NOT NULL,

            weed_density REAL NOT NULL,

            severity TEXT NOT NULL,

            detection_image TEXT,

            created_at TEXT NOT NULL

        )
        """
    )

    connection.commit()

    connection.close()

    print(
        "Database initialized successfully!"
    )


# Initialize database
init_database()


# =========================================================
# HOME ROUTE
# =========================================================

@app.route("/")
def home():

    return jsonify({

        "message":
            "AgriVision AI Backend is Running!",

        "status":
            "success"

    })


# =========================================================
# SERVE DETECTED IMAGES
# =========================================================

@app.route(
    "/results/detections/<path:filename>"
)
def serve_detection(filename):

    return send_from_directory(
        DETECTION_FOLDER,
        filename
    )


# =========================================================
# PREDICT / ANALYZE IMAGE
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # -----------------------------------------------------
    # CHECK IMAGE
    # -----------------------------------------------------

    if "image" not in request.files:

        return jsonify({

            "error":
                "No image uploaded"

        }), 400


    image = request.files["image"]


    if image.filename == "":

        return jsonify({

            "error":
                "No image selected"

        }), 400


    # -----------------------------------------------------
    # CHECK IMAGE TYPE
    # -----------------------------------------------------

    if not allowed_file(image.filename):

        return jsonify({

            "error":
                "Only JPG, JPEG and PNG images are allowed"

        }), 400


    # -----------------------------------------------------
    # SECURE FILE NAME
    # -----------------------------------------------------

    filename = secure_filename(
        image.filename
    )


    # -----------------------------------------------------
    # SAVE UPLOADED IMAGE
    # -----------------------------------------------------

    image_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    image.save(
        image_path
    )


    # -----------------------------------------------------
    # RUN YOLO MODEL
    # -----------------------------------------------------

    try:

        results = model.predict(

            source=image_path,

            conf=0.25,

            save=True,

            project=RESULT_FOLDER,

            name="detections",

            exist_ok=True,

            verbose=False

        )

    except Exception as error:

        return jsonify({

            "error":
                "Model prediction failed: "
                + str(error)

        }), 500


    # -----------------------------------------------------
    # GET FIRST RESULT
    # -----------------------------------------------------

    result = results[0]


    detections = []


    # -----------------------------------------------------
    # EXTRACT DETECTIONS
    # -----------------------------------------------------

    for box in result.boxes:

        class_id = int(
            box.cls.item()
        )

        confidence = float(
            box.conf.item()
        )

        coordinates = (
            box.xyxy[0].tolist()
        )

        x1 = coordinates[0]
        y1 = coordinates[1]
        x2 = coordinates[2]
        y2 = coordinates[3]


        # Class mapping
        #
        # 0 = Crop
        # 1 = Weed

        if class_id == 0:

            class_name = "Crop"

        elif class_id == 1:

            class_name = "Weed"

        else:

            class_name = "Unknown"


        detections.append({

            "class":
                class_name,

            "confidence":
                round(
                    confidence,
                    2
                ),

            "x1":
                round(
                    x1,
                    2
                ),

            "y1":
                round(
                    y1,
                    2
                ),

            "x2":
                round(
                    x2,
                    2
                ),

            "y2":
                round(
                    y2,
                    2
                )

        })


    # -----------------------------------------------------
    # COUNT CROP AND WEED
    # -----------------------------------------------------

    crop_count = int(
        (
            result.boxes.cls == 0
        )
        .sum()
        .item()
    )


    weed_count = int(
        (
            result.boxes.cls == 1
        )
        .sum()
        .item()
    )


    total_objects = (
        crop_count +
        weed_count
    )


    # -----------------------------------------------------
    # CALCULATE WEED DENSITY
    # -----------------------------------------------------

    if total_objects > 0:

        weed_density = (
            weed_count /
            total_objects
        ) * 100

    else:

        weed_density = 0


    weed_density = round(
        weed_density,
        2
    )


    # -----------------------------------------------------
    # CALCULATE SEVERITY
    #
    # Project thresholds:
    # < 5%       = Low
    # 5% - 20%   = Medium
    # > 20%      = High
    # -----------------------------------------------------

    if weed_density < 5:

        severity = "Low"

    elif weed_density <= 20:

        severity = "Medium"

    else:

        severity = "High"


    # -----------------------------------------------------
    # DETECTED IMAGE URL
    # -----------------------------------------------------

    detection_image = (
        "/results/detections/"
        + filename
    )


    # -----------------------------------------------------
    # SAVE ANALYSIS TO DATABASE
    # -----------------------------------------------------

    try:

        connection = sqlite3.connect(
            DATABASE
        )

        cursor = connection.cursor()


        created_at = (
            datetime.now()
            .strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )


        cursor.execute(
            """
            INSERT INTO detection_history
            (
                image_name,
                crop_count,
                weed_count,
                weed_density,
                severity,
                detection_image,
                created_at
            )

            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,

            (
                filename,

                crop_count,

                weed_count,

                weed_density,

                severity,

                detection_image,

                created_at
            )
        )


        connection.commit()

        connection.close()


    except Exception as error:

        print(
            "Database error:",
            error
        )


    # -----------------------------------------------------
    # SEND RESULT TO FRONTEND
    # -----------------------------------------------------

    return jsonify({

        "crop_count":
            crop_count,

        "weed_count":
            weed_count,

        "weed_density":
            weed_density,

        "severity":
            severity,

        "detected_image":
            detection_image,

        "detections":
            detections

    })


# =========================================================
# GET DETECTION HISTORY
# =========================================================

@app.route(
    "/history",
    methods=["GET"]
)
def get_history():

    try:

        connection = sqlite3.connect(
            DATABASE
        )

        connection.row_factory = (
            sqlite3.Row
        )

        cursor = connection.cursor()


        cursor.execute(
            """
            SELECT

                id,

                image_name,

                crop_count,

                weed_count,

                weed_density,

                severity,

                detection_image,

                created_at

            FROM detection_history

            ORDER BY id DESC
            """
        )


        rows = cursor.fetchall()

        connection.close()


        history = []


        for row in rows:

            history.append({

                "id":
                    row["id"],

                "image_name":
                    row["image_name"],

                "crop_count":
                    row["crop_count"],

                "weed_count":
                    row["weed_count"],

                "weed_density":
                    row["weed_density"],

                "severity":
                    row["severity"],

                "detection_image":
                    row["detection_image"],

                "created_at":
                    row["created_at"]

            })


        return jsonify(
            history
        )


    except Exception as error:

        return jsonify({

            "error":
                "Could not load history: "
                + str(error)

        }), 500


# =========================================================
# DELETE ONE HISTORY RECORD
# =========================================================

@app.route(
    "/history/<int:history_id>",
    methods=["DELETE"]
)
def delete_history(history_id):

    try:

        connection = sqlite3.connect(
            DATABASE
        )

        cursor = connection.cursor()


        cursor.execute(
            """
            DELETE FROM detection_history

            WHERE id = ?
            """,

            (
                history_id,
            )
        )


        deleted_rows = (
            cursor.rowcount
        )


        connection.commit()

        connection.close()


        if deleted_rows == 0:

            return jsonify({

                "error":
                    "History record not found"

            }), 404


        return jsonify({

            "message":
                "History deleted successfully"

        })


    except Exception as error:

        return jsonify({

            "error":
                "Could not delete history: "
                + str(error)

        }), 500


# =========================================================
# DELETE ALL HISTORY
# =========================================================

@app.route(
    "/history",
    methods=["DELETE"]
)
def delete_all_history():

    try:

        connection = sqlite3.connect(
            DATABASE
        )

        cursor = connection.cursor()


        cursor.execute(
            """
            DELETE FROM detection_history
            """
        )


        connection.commit()

        connection.close()


        return jsonify({

            "message":
                "All detection history deleted successfully"

        })


    except Exception as error:

        return jsonify({

            "error":
                "Could not delete history: "
                + str(error)

        }), 500


# =========================================================
# RUN FLASK SERVER
# =========================================================

if __name__ == "__main__":

    print("")
    print(
        "=========================================="
    )

    print(
        "       AGRIVISION AI BACKEND"
    )

    print(
        "=========================================="
    )

    print(
        "Server: http://127.0.0.1:5000"
    )

    print(
        "Prediction API: /predict"
    )

    print(
        "History API: /history"
    )

    print(
        "=========================================="
    )

    print("")


    # Render provides PORT environment variable.
    # Local testing uses 5000.

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )