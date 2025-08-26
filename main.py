from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import json
from typing import List
from io import BytesIO
import tika
import re
from tika import parser as p
from PIL import Image, ImageDraw, ImageFont
import mimetypes
import argparse
parser = argparse.ArgumentParser(description='Process environment profile.')
parser.add_argument('--port', type=int, help='Port to use for environment setup')


def run_trayicon():
    from pystray import MenuItem as item
    import pystray
    import time
    from PIL import Image
    import threading
    from pystray import Menu as menu
    import sys
    import os

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    env_file_path = os.path.join(os.path.dirname(__file__), '..', 'environment.txt')

    import tkinter as tk
    def show_notifiGPU(message):
        notification.notify(
            title='Notification',
            message=message,
            timeout=1
        )

    def versionOcr():
        logger.info(f"{current_time} - Application version")

    def log_action():
        logger.info(f"{current_time} - Go to Log Folder")
        subprocess.Popen(['explorer', os.path.abspath(log_folder_path)]) 
        # popen(['explorer', os.path.abspath(log_folder_path)])
    def update_app():
        import velopack
        logger.info(f"{current_time} - Check Application Update")
        manager = velopack.UpdateManager("https://the.place/you-host/updates")

        update_info = manager.check_for_updates()
        if not update_info:
            return # no updates available

        # Download the updates, optionally providing progress callbacks
        manager.download_updates(update_info)

        # Apply the update and restart the app
        manager.apply_updates_and_restart(update_info)
    from plyer import notification
    def show_notification1(message):
        notification.notify(
            title='Notification',
            message=message,
            timeout=10  # Notification duration in seconds
        )
        logger.info(f"{current_time} - Check Dictionary Update, Do Update")
     
    def on_activateDict1():
        threading.Thread(target=lambda: show_notification1("There is a new entry in the dictionary. The application will stop and turn on again.")).start()

    def show_notification2(message):
        notification.notify(
            title='Notification',
            message=message,
            timeout=10  # Notification duration in seconds
        )
    def on_activateDict2():
        threading.Thread(target=lambda: show_notification2("The dictionary is the latest version")).start()

    def exit_action(icon, item):
        logger.info(f"{current_time} - Application OCR Stopped")
        os._exit(0)        

    def update_env_variable(key, value, env_file_path=env_file_path):
        with open(env_file_path, "r") as file:
            lines = file.readlines()

        with open(env_file_path, "w") as file:
            for line in lines:
                if line.startswith(key + "="):
                    file.write(f"{key}={value}\n")
                else:
                    file.write(line)

    def read_env_variable(key, env_file_path=env_file_path):
        with open(env_file_path, "r") as file:
            for line in file:
                if line.startswith(key + "="):
                    return line.strip().split('=')[1]
        return None

    def CPU_mode(icon, item):
        logger.info(f"{current_time} - Change to CPU mode")
        if read_env_variable("SET_GPU") == "GPU":
            # show_notification(icon, item)
            # update_env_variable("SET_GPU", "CPU")
            # exit_action(icon, item)
            show_notifiGPU("The application will stop and turn on again")
            time.sleep(10)
            update_env_variable("SET_GPU", "CPU")
            exit_action(icon, item)

    sep = pystray.Menu.SEPARATOR
    disabled_item = pystray.MenuItem('iCapture-1.0', versionOcr, enabled=False)

    def setup_program(icon):
        for i in range(5):
            time.sleep(1)
        icon.update_menu_running()

    menu_preparing = menu = (
        item('Preparing', lambda: None, enabled=False),
        disabled_item,
        sep,
        item('Open Log', log_action),
        item('Exit', exit_action),
    )

    menu_running = menu = (
        item('Running', lambda: None, enabled=False),
        disabled_item,
        sep,
        item('Open Log', log_action),
        item('Check Application Update', check_app_update),
        item('Exit', exit_action)
    )
    
    def create_image():
        if getattr(sys, 'frozen', False):
            wd = sys._MEIPASS
        else:
            wd = ''  
        # Load the icon from file
        icon_path = os.path.join(wd, "icon.png")
        return Image.open(icon_path)

    icon = pystray.Icon("name", create_image(), "ISWITCH-iCapture",menu_preparing)

    def update_menu_running():
        icon.menu = menu_running
        icon.update_menu()

    icon.update_menu_running = update_menu_running
    threading.Thread(target=setup_program, args=(icon,)).start()

    icon.run_detached()

args, unknown = parser.parse_known_args()

port = args.port if args.port is not None else 8000

try:
    from pdf2image import convert_from_bytes
except ImportError:
    convert_from_bytes = None

from logger import logger
from internvl_utils import (
    ask_internvl_ai,
    add_default_result_fields,
    set_internvl_model
)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

@app.post("/ocr-internvl/predict")
async def predict_ocr(listDocumentType: str = Form(...), files: list[UploadFile] = File(...)):
    list_document_type_json = json.loads(listDocumentType)
    print(f"[api :: predict] .... ")
    doc_types = []
    for doc_type in list_document_type_json.get("listDocType", []):
        if doc_type.get("active") is False:
            continue
        filtered_metadata = [m for m in doc_type.get("metadata", []) if m.get("active", True)]
        label_name_map = {m.get("label", ""): m.get("name", "") for m in filtered_metadata}
        doc_types.append({
            "label": doc_type.get("label", ""),
            "id_doctype": doc_type.get("id_doctype", {}),
            "metadata": [m.get("label", "") for m in filtered_metadata],
            "metadata_name": [m.get("name", "") for m in filtered_metadata],
            "label_name_map": label_name_map
        })

    pixels = []
    for image in files:
        image_data = await image.read()
        pixels.append(image_data)

    question = (
        "From the following document types and their fields, select the best match for the image and extract the values for each field.\n"
        "Return the result as a single string in the following format:\n"
        "'Label1=>Value, Label2=>Value, ... || doc_type_index'\n"
        "Use the exact label as shown. If a value is not found, leave it empty after '=>'.\n"
        "The doc_type_index is the index of the selected document type (starting from 0), separated by '||'.\n"
        "Document types and fields:\n"
    )
    for idx, doc_type in enumerate(doc_types):
        question += f"{idx}. {doc_type['label']}: {', '.join(doc_type['metadata'])}\n"

    response = ask_internvl_ai(pixels, question, max_num=12)

    try:
        if isinstance(response, bytes):
            response = response.decode()
        if "||" not in response:
            raise ValueError("Invalid response format")
        mapping_part, idx_part = response.rsplit("||", 1)
        doc_type_index = int(idx_part.strip())
        selected_doc_type = doc_types[doc_type_index]
        label_name_map = selected_doc_type["label_name_map"]
        ocr_mapping_result = []
        for pair in mapping_part.strip().split(","):
            if "=>" not in pair:
                continue
            label, value = pair.split("=>", 1)
            label = label.strip()
            value = value.strip()
            if label in label_name_map:
                ocr_mapping_result.append({
                    "label": label,
                    "name": label_name_map[label],
                    "value": value
                })
        result = {
            "ocrMappingResult": ocr_mapping_result,
            "id_doctype": selected_doc_type["id_doctype"]
        }
    except Exception as e:
        logger.error(f"Failed to parse AI response: {e}")
        result = {}

    result = add_default_result_fields(result)
    return result

@app.post("/ocr-internvl/mapping")
async def ocr_internvl_mapping(
    documentType: str = Form(...),
    files: list[UploadFile] = File(...)
):
    try:
        print(f"tipe: {type(documentType)}")
        if isinstance(documentType,str):
            document_type_json = json.loads(documentType)
        else:
            document_type_json = documentType
    except Exception as err:
        print(f"Cannot Load json str document type:{err}")
    print(f"[ocr_internvl_mapping] ....\n{documentType} ")
    # print(f"[api :: mapping] .... ")
    filtered_metadata = [m for m in document_type_json.get("metadata", []) if m.get("active", True)]
    

    metadata_labels = [str(m.get("label", "")).rstrip() for m in filtered_metadata]
    
    label_name_map = {str(m.get("label", "")).rstrip(): m.get("name", "") for m in filtered_metadata}

    pixels = []
    for image in files:
        is_image = True
        # cek apakah file adalah pdf
        filename = image.filename.lower()
        content_type = image.content_type or mimetypes.guess_type(filename)[0]
        if filename.endswith('.pdf') or (content_type and 'pdf' in content_type):
            # hanya ambil file pdf pertama
            await image.seek(0)
            pdf_bytes = await image.read()
            if convert_from_bytes is None:
                logger.error("pdf2image is not installed")
                continue
            try:
                pil_images: List[Image.Image] = convert_from_bytes(pdf_bytes, first_page=1, last_page=1)
                if pil_images:
                    buf = BytesIO()
                    pil_images[0].save(buf, format="PNG")
                    pixels.append(buf.getvalue())
            except Exception as e:
                logger.error(f"Failed to convert PDF to image: {e}")
            break  # hanya proses pdf pertama
        elif filename.endswith((".xlsx",".xls",".doc",".docx",".ppt",".pptx",".odt",".ods",".odp")):
            await image.seek(0)
            docs_byte = await image.read()
            results = p.from_buffer(docs_byte)
            content = results['content']
            cleaned_content = re.sub(r'\n{3,}', '\n\n', content)
            if filename.endswith(('.xlsx', '.xls','.ods')):
                cleaned_content = re.sub(r'\n{4,}', '\n', cleaned_content)  # More aggressive newline reduction
            elif filename.endswith(('.docx', '.doc', '.ppt', '.pptx','.odt','.odp')):
                cleaned_content = re.sub(r'\n\.{2,}', '', cleaned_content)  # Remove dot leaders
            data = cleaned_content
            is_image = False
        else:
            await image.seek(0)
            image_data = await image.read()
            pixels.append(image_data)
    
    if is_image:
        question = (
            "<image>\n"
            "Extract the values for the following fields from the image.\n"
            "Return the result as a single string in the following format:\n"
            "'Label1=>Value, Label2=>Value, ...'\n"
            "Use the exact label as shown. If a value is not found, leave it empty after '=>'.\n"
            "Fields: " + ", ".join(metadata_labels)
        )
    else:
        question = (
            f"{data}\n"
            "Extract the values for the following fields from the data.\n"
            "Return the result as a single string in the following format:\n"
            "'Label1=>Value, Label2=>Value, ...'\n"
            "Use the exact label as shown. If a value is not found, leave it empty after '=>'.\n"
            "Fields: " + ", ".join(metadata_labels)
        )
    response = ask_internvl_ai(pixels, question, max_num=12)

    try:
        if isinstance(response, bytes):
            response = response.decode()
        ocr_mapping_result = []
        # print(f"response after:{response}")
        for pair in response.strip().split(","):
            if "=>" not in pair:
                continue
            # print(f"pair:{pair} - label map :{label_name_map}")
            pair_arr = pair.split("=>")
            # label, value = pair.split("=>", 1)
            label = pair_arr[0].strip()
            value = pair_arr[1].strip()
            if label in label_name_map:
                ocr_mapping_result.append({
                    "label": label,
                    "name": label_name_map[label],
                    "value": value
                })
        result = {
            "ocrMappingResult": ocr_mapping_result,
            "id_doctype": document_type_json.get("id_doctype", {})
        }
        #print("return --- result:")
        #print(result)
    except Exception as e:
        logger.error(f"Failed to parse AI response: {e}")
        result = {}

    result = add_default_result_fields(result)
    print(f"Result:{result}")
    return result

@app.post("/ocr-internvl/question")
async def ocr_internvl_question(
    prompt: str = Form(...),
    files: list[UploadFile] = File(None)
):
    image_datas = []
    if files:
        for image in files:
            image_datas.append(await image.read())
    else:
        image_datas = None
    response = ask_internvl_ai(image_datas, prompt, max_num=12)
    return {"response": response}

@app.post("/ocr-internvl/set-model")
async def set_model(
    model_path: str = Form(...),
    quant_mode: str = Form("4bit")  # default tetap 4bit agar backward compatible
):
    try:
        set_internvl_model(model_path, quant_mode)
        return {"success": True, "message": f"Model changed to {model_path} with quant_mode={quant_mode}"}
    except Exception as e:
        logger.error(f"Failed to set model: {e}")
        return {
            "success": False,
            "message": "Failed to set model",
            "error": str(e)
        }

def main():
    import uvicorn
    logger.info("Starting the InternVL API server")
    run_trayicon()
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False, log_level="info")

if __name__ == "__main__":
    print("Hi")
    main()