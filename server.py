import os
import threading
import requests
import json
import pandas as pd
from pathlib import Path
import re
from flask import Flask, jsonify, request
from main import main_combined as fetch_link_main
from process_data import main_combined as process_data
import traceback

app = Flask(__name__)

SERVICE_NAME = "Data_Collection"
BASE_PATH = os.getenv("BASE_PATH") or "/nas_data/"

if not (os.path.isdir(BASE_PATH) and os.access(BASE_PATH, os.R_OK)):
    BASE_PATH = "/app/"


def run_pipeline_and_callback(callback_url, func, *func_args, **func_kwargs):
    """
    This function runs in the background thread.
    It contains your complete task logic and sends a callback to Webhook Url when finished.
    """
    payload = {}

    try:
        # <your-main-program>
        # result = extract_raw_data(base_path)
        result = func(*func_args, **func_kwargs)

        # Organize your task's output result into the 'result' field
        payload = {
            "status": "success",
            "service": SERVICE_NAME,
            "result": result
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        payload = {
            "status": "failed",
            "service": SERVICE_NAME,
            "error": str(e)
        }

    finally:
        if callback_url:
            try:
                requests.post(callback_url, json=payload, timeout=20)
                print("Callback sent successfully.")
            except requests.RequestException as req_e:
                print(f"Failed to send callback: {req_e}")
        else:
            print("No callback_url provided. Skipping callback.")


@app.route('/run', methods=['POST'])
def trigger_run():
    """
    The entry point for the web service. It receives the launch request.
    """
    data = request.get_json()
    if not data or 'callback_url' not in data:
        return jsonify({"error": "Missing 'callback_url' in request body"}), 400
    callback_url = data['callback_url']
    month = data['month'] if "month" in data else None

    thread = threading.Thread(
        target=run_pipeline_and_callback,
        args=(callback_url,
              fetch_link_main,
              BASE_PATH,
              month
              )
    )
    thread.daemon = True
    thread.start()

    response = {
        "message": "Task accepted and is running in the background.",
        "service": SERVICE_NAME
    }
    return jsonify(response), 202


@app.route('/process_data', methods=['POST'])
def trigger_process_data():
    """
    The entry point for the web service. It receives the launch request.
    """
    data = request.get_json()
    if not data or 'callback_url' not in data:
        return jsonify({"error": "Missing 'callback_url' in request body"}), 400
    callback_url = data['callback_url']

    thread = threading.Thread(
        target=run_pipeline_and_callback,
        args=(callback_url,
              process_data,
              BASE_PATH
              )
    )
    thread.daemon = True
    thread.start()

    response = {
        "message": "Task accepted and is running in the background.",
        "service": SERVICE_NAME
    }
    return jsonify(response), 202


@app.route('/get_last_import_date', methods=['GET'])
def get_last_import_date():
    with open(os.path.join(BASE_PATH, "update_masterlist.json"),
              "r", encoding="utf-8") as f:
        config = json.load(f)
    excel_config = config["excel_config"]
    excel_path = excel_config["excel_file"]
    sheet_configs = excel_config["sheets"]
    excel_data = pd.read_excel(excel_path, sheet_name=None, engine="openpyxl")
    res = {}
    for sheet_config in sheet_configs:
        sheet_name = sheet_config["sheet_name"]
        if not sheet_name.startswith("LLM_"):
            master_df = excel_data[sheet_name]
            import_date = master_df["import_date"]
            res[sheet_name] = import_date.max().strftime("%Y-%m-%d")

    response = {
        "status": "success",
        "service": SERVICE_NAME,
        "result": res
    }
    return jsonify(response), 200


def find_latest_date_dir(parent_dir: str | Path) -> Path | None:
    parent = Path(parent_dir)
    date_regex = re.compile(r'^\d{8}$')
    candidates = [
        p for p in parent.iterdir()
        if p.is_dir() and date_regex.match(p.name)
    ]
    return max(candidates, key=lambda p: p.name).name


@app.route('/get_prev_ver', methods=['GET'])
def get_prev_ver():
    eng_path = os.path.join(BASE_PATH, "..", "output", "English")
    chi_path = os.path.join(BASE_PATH, "..", "output", "Chinese")

    eng_prev_ver = find_latest_date_dir(eng_path)
    chi_prev_ver = find_latest_date_dir(chi_path)

    response = {
        "status": "success",
        "service": SERVICE_NAME,
        "result": {
            "eng": str(eng_prev_ver),
            "chi": str(chi_prev_ver)
        }
    }
    return jsonify(response), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
