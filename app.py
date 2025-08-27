import sys
import os
import subprocess
import sqlite3
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from ultralytics import YOLO
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QFileDialog, QTextEdit,
                             QProgressBar, QMessageBox, QGridLayout, QGroupBox, 
                             QScrollArea, QLineEdit, QComboBox, QDateEdit, QRadioButton,
                             QButtonGroup, QFrame, QSplitter, QListWidget, QListWidgetItem,
                             QTabWidget, QTextBrowser, QFormLayout, QSpacerItem, QSizePolicy,
                             QMenu, QAction)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDate, QTimer, QSize
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor, QIcon, QPainter
import numpy as np
import cv2
from datetime import datetime, date

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "Model")

# Portable environment bootstrap: prefer bundled Python at ./CCCD/python.exe
def _maybe_relaunch_with_local_python():
    try:
        if os.name != 'nt':
            return
        local_env_dir = os.path.join(SCRIPT_DIR, 'CCCD')
        candidates = [
            os.path.join(local_env_dir, 'pythonw.exe'),
            os.path.join(local_env_dir, 'python.exe'),
        ]
        local_python = next((p for p in candidates if os.path.exists(p)), None)
        if not local_python:
            return
        current = os.path.normcase(os.path.abspath(sys.executable))
        target = os.path.normcase(os.path.abspath(local_python))
        if current == target:
            return
        # Relaunch the app with the local Python, preserving args
        os.chdir(SCRIPT_DIR)
        cmd = [local_python, os.path.abspath(__file__), *sys.argv[1:]]
        subprocess.Popen(cmd)
        sys.exit(0)
    except Exception:
        # Fail open: continue with current interpreter if anything goes wrong
        pass

_maybe_relaunch_with_local_python()

class DatabaseManager:
    def __init__(self, db_path="cccd_database.sqlite"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create CCCD records table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cccd_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ho_ten TEXT NOT NULL,
            gioi_tinh TEXT,
            so_cccd TEXT UNIQUE NOT NULL,
            ngay_sinh DATE,
            ngay_cap DATE,
            ngay_het_han DATE,
            que_quan TEXT,
            dan_toc TEXT,
            quoc_tich TEXT DEFAULT 'Việt Nam',
            noi_thuong_tru TEXT,
            image_front_path TEXT,
            image_back_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_record(self, data):
        """Save or update CCCD record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if record exists
            cursor.execute("SELECT id FROM cccd_records WHERE so_cccd = ?", (data['so_cccd'],))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing record
                cursor.execute('''
                UPDATE cccd_records SET 
                    ho_ten=?, gioi_tinh=?, ngay_sinh=?, ngay_cap=?, ngay_het_han=?,
                    que_quan=?, dan_toc=?, quoc_tich=?, noi_thuong_tru=?,
                    image_front_path=?, image_back_path=?, updated_at=CURRENT_TIMESTAMP
                WHERE so_cccd=?
                ''', (
                    data['ho_ten'], data['gioi_tinh'], data['ngay_sinh'], data['ngay_cap'],
                    data['ngay_het_han'], data['que_quan'], data['dan_toc'], data['quoc_tich'],
                    data['noi_thuong_tru'], data['image_front_path'], data['image_back_path'],
                    data['so_cccd']
                ))
            else:
                # Insert new record
                cursor.execute('''
                INSERT INTO cccd_records 
                (ho_ten, gioi_tinh, so_cccd, ngay_sinh, ngay_cap, ngay_het_han,
                 que_quan, dan_toc, quoc_tich, noi_thuong_tru, image_front_path, image_back_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data['ho_ten'], data['gioi_tinh'], data['so_cccd'], data['ngay_sinh'],
                    data['ngay_cap'], data['ngay_het_han'], data['que_quan'], data['dan_toc'],
                    data['quoc_tich'], data['noi_thuong_tru'], data['image_front_path'], data['image_back_path']
                ))
            
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            print(f"Database error: {e}")
            return False
        finally:
            conn.close()
    
    def search_by_cccd(self, cccd_number):
        """Search record by CCCD number"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM cccd_records WHERE so_cccd = ?", (cccd_number,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            columns = ['id', 'ho_ten', 'gioi_tinh', 'so_cccd', 'ngay_sinh', 'ngay_cap', 
                      'ngay_het_han', 'que_quan', 'dan_toc', 'quoc_tich', 'noi_thuong_tru',
                      'image_front_path', 'image_back_path', 'created_at', 'updated_at']
            return dict(zip(columns, result))
        return None
    
    def get_all_records(self):
        """Get all records with basic info"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT so_cccd, ho_ten, ngay_sinh FROM cccd_records ORDER BY updated_at DESC")
        results = cursor.fetchall()
        conn.close()
        
        return results

    def delete_by_cccd(self, cccd_number):
        """Delete a record by CCCD number"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM cccd_records WHERE so_cccd = ?", (cccd_number,))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            conn.rollback()
            print(f"Database delete error: {e}")
            return False
        finally:
            conn.close()

class DetectionThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, image_paths, model_path):
        super().__init__()
        self.image_paths = image_paths
        self.model_path = model_path
        
    def run(self):
        try:
            model = YOLO(self.model_path)
            all_results = {}
            
            for idx, image_path in enumerate(self.image_paths):
                if image_path is None:
                    continue
                    
                self.progress.emit(int((idx / len(self.image_paths)) * 50))
                
                results = model.predict(source=image_path, conf=0.5, imgsz=640, save=False, verbose=False)
                
                detected_info = {}
                class_groups = {}
                
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        img = cv2.imread(image_path)
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            class_names = {
                                0: "current_place", 1: "dob", 2: "expire_date",
                                5: "gender", 6: "id", 7: "issue_date", 
                                8: "name", 9: "nationality", 10: "origin_place"
                            }
                            
                            if cls_id in [3, 4, 11]:
                                continue
                                
                            field_name = class_names.get(cls_id, f"unknown_{cls_id}")
                            
                            if field_name not in class_groups:
                                class_groups[field_name] = []
                            
                            class_groups[field_name].append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': conf
                            })
                        
                        for field_name, detections in class_groups.items():
                            if len(detections) == 1:
                                detected_info[field_name] = detections[0]
                            else:
                                merged_bbox = self.merge_bounding_boxes([d['bbox'] for d in detections])
                                avg_confidence = sum(d['confidence'] for d in detections) / len(detections)
                                
                                detected_info[field_name] = {
                                    'bbox': merged_bbox,
                                    'confidence': avg_confidence,
                                    'original_detections': detections
                                }
                        
                        for field_name, info in detected_info.items():
                            bbox = info['bbox']
                            conf = info['confidence']
                            cv2.rectangle(img_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                            label = f"{field_name}: {conf:.2f}"
                            cv2.putText(img_rgb, label, (bbox[0], bbox[1]-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        annotated_image_path = f"temp_annotated_{idx}.jpg"
                        cv2.imwrite(annotated_image_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                
                all_results[f"image_{idx}"] = {
                    'detected_info': detected_info,
                    'annotated_path': annotated_image_path,
                    'original_path': image_path
                }
            
            self.finished.emit(all_results)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def merge_bounding_boxes(self, bboxes):
        if not bboxes:
            return [0, 0, 0, 0]
        if len(bboxes) == 1:
            return bboxes[0]
        
        min_x = min(bbox[0] for bbox in bboxes)
        min_y = min(bbox[1] for bbox in bboxes)
        max_x = max(bbox[2] for bbox in bboxes)
        max_y = max(bbox[3] for bbox in bboxes)
        
        return [min_x, min_y, max_x, max_y]

class OCRThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, all_detected_results):
        super().__init__()
        self.all_detected_results = all_detected_results
        
    def run(self):
        try:
            model_path = MODEL_DIR
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path not found: {model_path}")
            
            try:
                model = AutoModel.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                    trust_remote_code=True, use_flash_attn=False, local_files_only=True
                ).eval()
                
                if torch.cuda.is_available():
                    model = model.cuda()
                else:
                    model = model.cpu()
                    
            except Exception as e:
                model = AutoModel.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                    trust_remote_code=True, local_files_only=True
                ).eval()
                
                if torch.cuda.is_available():
                    model = model.cuda()
                else:
                    model = model.cpu()
            
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, local_files_only=True)
            
            all_ocr_results = {}
            total_regions = sum(len(result['detected_info']) for result in self.all_detected_results.values())
            processed_regions = 0
            
            for image_key, image_result in self.all_detected_results.items():
                image_path = image_result['original_path']
                detected_regions = image_result['detected_info']
                
                if not detected_regions:
                    continue
                
                image = Image.open(image_path).convert('RGB')
                
                for field_name, region_info in detected_regions.items():
                    processed_regions += 1
                    self.progress.emit(int((processed_regions / total_regions) * 100))
                    
                    if 'original_detections' in region_info:
                        individual_texts = []
                        
                        for det in region_info['original_detections']:
                            x1, y1, x2, y2 = det['bbox']
                            cropped_img = image.crop((x1, y1, x2, y2))
                            temp_path = f"temp_{field_name}_{image_key}_{len(individual_texts)}.jpg"
                            cropped_img.save(temp_path)
                            
                            try:
                                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                                pixel_values = self.load_image(temp_path, max_num=6)
                                
                                if device == 'cuda':
                                    pixel_values = pixel_values.to(torch.bfloat16).cuda()
                                else:
                                    pixel_values = pixel_values.to(torch.float32).cpu()
                                
                                prompt = self.get_prompt(field_name)
                                generation_config = dict(
                                    max_new_tokens=256 if field_name in ["current_place", "origin_place"] else 128,
                                    do_sample=False, num_beams=3, repetition_penalty=3.5,
                                )
                                
                                response = model.chat(tokenizer, pixel_values, prompt, generation_config)
                                individual_texts.append(response.strip())
                                
                            except Exception as e:
                                individual_texts.append("")
                            finally:
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                        
                        if individual_texts:
                            valid_texts = [text for text in individual_texts if text.strip()]
                            if valid_texts:
                                all_ocr_results[field_name] = ", ".join(valid_texts)
                            else:
                                all_ocr_results[field_name] = ""
                        else:
                            all_ocr_results[field_name] = ""
                    
                    else:
                        x1, y1, x2, y2 = region_info['bbox']
                        cropped_img = image.crop((x1, y1, x2, y2))
                        temp_path = f"temp_{field_name}_{image_key}.jpg"
                        cropped_img.save(temp_path)
                        
                        try:
                            device = 'cuda' if torch.cuda.is_available() else 'cpu'
                            pixel_values = self.load_image(temp_path, max_num=6)
                            
                            if device == 'cuda':
                                pixel_values = pixel_values.to(torch.bfloat16).cuda()
                            else:
                                pixel_values = pixel_values.to(torch.float32).cpu()
                            
                            prompt = self.get_prompt(field_name)
                            generation_config = dict(
                                max_new_tokens=256 if field_name in ["current_place", "origin_place"] else 128,
                                do_sample=False, num_beams=3, repetition_penalty=3.5,
                            )
                            
                            response = model.chat(tokenizer, pixel_values, prompt, generation_config)
                            all_ocr_results[field_name] = response.strip()
                            
                        except Exception as e:
                            all_ocr_results[field_name] = ""
                        finally:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
            
            self.finished.emit(all_ocr_results)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def get_prompt(self, field_name):
        prompts = {
            "current_place": """<image>
Nhiệm vụ: Đọc văn bản trên ảnh CCCD và CHỈ trích xuất giá trị của trường "Nơi thường trú" (Place of residence).
YÊU CẦU:
- Trả về NGUYÊN VĂN đúng như trên thẻ: giữ nguyên dấu, chữ hoa/thường, dấu phẩy/chấm, khoảng trắng.
- KHÔNG chuẩn hoá/viết lại, KHÔNG rút gọn, KHÔNG thêm/bớt từ nào.
- Nếu địa chỉ xuống dòng, hãy NỐI các dòng bằng MỘT khoảng trắng, không thêm ký tự khác.
- KHÔNG thêm nhãn, KHÔNG giải thích, KHÔNG thêm dấu ngoặc.
- Đọc đầy đủ tất cả các dòng của địa chỉ, không bỏ sót thông tin nào.

Chỉ trả về đúng chuỗi địa chỉ như trên thẻ.""",
            
            "dob": """<image>
Hãy đọc văn bản trên ảnh và chỉ trích xuất giá trị của trường "Ngày sinh" (Date of birth).
Trả về đúng định dạng ngày tháng năm, không thêm giải thích, không thêm nhãn.""",
            
            "expire_date": """<image>
Hãy đọc văn bản trên ảnh và chỉ trích xuất giá trị của trường "Ngày hết hạn" (Expiry date).
Trả về đúng định dạng ngày tháng năm, không thêm giải thích, không thêm nhãn.""",
            
            "gender": """<image>
Hãy đọc văn bản trên ảnh và chỉ trích xuất giá trị của trường "Giới tính" (Gender).
Trả về đúng giới tính (Nam/Nữ), không thêm giải thích, không thêm nhãn.""",
            
            "id": """<image>
Hãy đọc văn bản trên ảnh và chỉ trích xuất giá trị của trường "Số CCCD/CMND" (ID number).
Trả về đúng chuỗi số, không thêm giải thích, không thêm nhãn.""",
            
            "issue_date": """<image>
Hãy đọc văn bản trên ảnh và chỉ trích xuất giá trị của trường "Ngày cấp" (Issue date).
Trả về đúng định dạng ngày tháng năm, không thêm giải thích, không thêm nhãn.""",
            
            "name": """<image>
Hãy đọc văn bản trên ảnh và chỉ trích xuất giá trị của trường "Họ và tên" (Full name).
Trả về đúng tên đầy đủ, không thêm giải thích, không thêm nhãn.""",
            
            "nationality": """<image>
Hãy đọc văn bản trên ảnh và chỉ trích xuất giá trị của trường "Quốc tịch" (Nationality).
Trả về đúng quốc tịch, không thêm giải thích, không thêm nhãn.""",
            
            "origin_place": """<image>
Nhiệm vụ: Đọc văn bản trên ảnh CCCD và CHỈ trích xuất giá trị của trường "Quê quán" (Place of origin).
YÊU CẦU:
- Trả về NGUYÊN VĂN đúng như trên thẻ: giữ nguyên dấu, chữ hoa/thường, dấu phẩy/chấm, khoảng trắng.
- KHÔNG chuẩn hoá/viết lại, KHÔNG rút gọn, KHÔNG thêm/bớt địa danh.
- Nếu giá trị xuống dòng, NỐI các dòng bằng MỘT khoảng trắng.
- KHÔNG thêm nhãn, KHÔNG giải thích, KHÔNG thêm dấu ngoặc.

Chỉ trả về đúng chuỗi quê quán."""
        }
        
        return prompts.get(field_name, f"""<image>
Hãy đọc văn bản trên ảnh và trích xuất thông tin.
Trả về đúng nội dung, không thêm giải thích, không thêm nhãn.""")
    
    def load_image(self, image_file, input_size=448, max_num=6):
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(img) for img in images]
        return torch.stack(pixel_values)
    
    def build_transform(self, input_size):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform
    
    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images
    
    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

class EnhancedCCCDApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.db_manager = DatabaseManager()
        self.detection_thread = None
        self.ocr_thread = None
        self.all_detection_results = {}
        self.current_results = {}
        
        # Vietnamese ethnic groups
        self.ethnic_groups = [
            "Kinh", "Tày", "Thái", "Mường", "Khmer", "Hoa", "Nùng", "H'Mông", "Dao", "Gia Rai",
            "Ê Đê", "Ba Na", "Sán Chay", "Chăm", "Cơ Ho", "Xơ Đăng", "Sán Dìu", "Hrê", "Ra Glai",
            "Mnong", "Thổ", "Xtiêng", "Khmu", "Cơ Tu", "Gié Triêng", "Tà Ôi", "Mạ", "Co",
            "Chơ Ro", "Xinh Mun", "Hà Nhì", "Chu Ru", "Lào", "La Chí", "Phù Lá", "La Hủ",
            "Lự", "Ngái", "Chứt", "Lô Lô", "Mảng", "Pà Thẻn", "Cống", "Bố Y", "Si La",
            "Pu Péo", "Brâu", "Ơ Đu", "Rơ Măm"
        ]
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Document Management System - CCCD/CMND Processing')
        self.setGeometry(100, 100, 1800, 1100)  # Tăng kích thước cửa sổ
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                color: #495057;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:disabled {
                background-color: #6c757d;
                color: #adb5bd;
            }
            QLineEdit, QTextEdit, QComboBox, QDateEdit {
                border: 2px solid #ced4da;
                border-radius: 6px;
                padding: 8px 12px;
                background-color: white;
                font-size: 13px;
                min-height: 20px;
            }
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QDateEdit:focus {
                border-color: #007bff;
            }
            QLabel {
                color: #495057;
                font-size: 13px;
            }
            QRadioButton {
                font-size: 13px;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
            }
        """)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)  # Tăng khoảng cách giữa sidebar và main content
        main_layout.setContentsMargins(20, 20, 20, 20)  # Tăng margin
        
        # Create sidebar and main content
        sidebar = self.create_sidebar()
        main_content = self.create_main_content()
        
        # Add to main layout with better proportions
        main_layout.addWidget(sidebar, 1)  # Sidebar chiếm 1 phần
        main_layout.addWidget(main_content, 5)  # Main content chiếm 5 phần
        
        main_widget.setLayout(main_layout)
        
        # Initialize variables
        self.front_image_path = None
        self.back_image_path = None
        
    def create_sidebar(self):
        """Create left sidebar with menu items"""
        sidebar = QFrame()
        sidebar.setFrameStyle(QFrame.StyledPanel)
        sidebar.setMaximumWidth(320)  # Tăng chiều rộng sidebar
        sidebar.setMinimumWidth(280)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #343a40;
                color: white;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(15)  # Tăng khoảng cách giữa các phần tử
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title = QLabel("Document Management")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: white; padding: 25px; background-color: #495057; border-radius: 8px; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Menu items
        menu_items = [
            ("CMND/CCCD", True),
            ("Hộ chiếu", False),
            ("Bằng cấp", False),
            ("Chứng chỉ", False),
            ("CV", False)
        ]
        
        self.menu_buttons = []
        for item_name, is_active in menu_items:
            btn = QPushButton(item_name)
            btn.setCheckable(True)
            btn.setMinimumHeight(55)  # Tăng chiều cao nút
            
            if is_active:
                btn.setChecked(True)
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #28a745;
                        color: white;
                        text-align: left;
                        padding-left: 25px;
                        font-size: 14px;
                        border-radius: 8px;
                    }
                    QPushButton:checked {
                        background-color: #1e7e34;
                    }
                """)
                btn.clicked.connect(self.menu_item_clicked)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #6c757d;
                        color: white;
                        text-align: left;
                        padding-left: 25px;
                        font-size: 14px;
                        border-radius: 8px;
                    }
                    QPushButton:hover {
                        background-color: #5a6268;
                    }
                """)
                btn.clicked.connect(lambda: self.show_coming_soon(item_name))
            
            self.menu_buttons.append(btn)
            layout.addWidget(btn)
        
        # Quick search section
        layout.addWidget(QLabel(""))  # Spacer
        search_group = QGroupBox("Tìm kiếm nhanh")
        search_group.setStyleSheet("QGroupBox { color: white; border-color: #6c757d; }")
        search_layout = QVBoxLayout()
        search_layout.setSpacing(10)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Nhập số CCCD/CMND...")
        self.search_input.returnPressed.connect(self.search_record)
        
        search_btn = QPushButton("Tìm kiếm")
        search_btn.clicked.connect(self.search_record)
        
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(search_btn)
        search_group.setLayout(search_layout)
        layout.addWidget(search_group)
        
        # Recent records
        recent_group = QGroupBox("Hồ sơ gần đây")
        recent_group.setStyleSheet("QGroupBox { color: white; border-color: #6c757d; }")
        recent_layout = QVBoxLayout()
        
        self.recent_list = QListWidget()
        self.recent_list.setStyleSheet("QListWidget { background-color: white; color: black; border-radius: 6px; }")
        self.recent_list.setIconSize(QSize(128, 48))
        self.recent_list.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.recent_list.itemClicked.connect(self.load_recent_record)
        self.recent_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.recent_list.customContextMenuRequested.connect(self.show_recent_context_menu)
        
        recent_layout.addWidget(self.recent_list)
        recent_group.setLayout(recent_layout)
        layout.addWidget(recent_group)
        
        layout.addStretch()

        # Bọc layout sidebar trong scroll area để tránh đè nén giao diện
        inner_widget = QWidget()
        inner_widget.setLayout(layout)

        scroll_area = QScrollArea()
        scroll_area.setWidget(inner_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(scroll_area)
        sidebar.setLayout(container_layout)
        
        # Load recent records
        self.load_recent_records()
        
        return sidebar

    def show_recent_context_menu(self, pos):
        """Context menu for recent records list: delete record"""
        item = self.recent_list.itemAt(pos)
        if not item:
            return
        cccd = item.data(Qt.UserRole)

        menu = QMenu(self)
        delete_action = QAction("Xóa hồ sơ", self)
        menu.addAction(delete_action)

        action = menu.exec_(self.recent_list.mapToGlobal(pos))
        if action == delete_action:
            reply = QMessageBox.question(
                self,
                "Xác nhận xóa",
                f"Bạn chắc chắn muốn xóa hồ sơ {cccd}?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                if self.db_manager.delete_by_cccd(cccd):
                    self.load_recent_records()
                    QMessageBox.information(self, "Đã xóa", "Hồ sơ đã được xóa.")
                else:
                    QMessageBox.critical(self, "Lỗi", "Không thể xóa hồ sơ.")
    
    def create_main_content(self):
        """Create main content area"""
        main_content = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(20)  # Tăng khoảng cách giữa các section
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Guidelines section
        guidelines = self.create_guidelines_section()
        layout.addWidget(guidelines, 1)
        
        # Image upload and form section
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setChildrenCollapsible(False)
        
        # Left: Image upload
        image_section = self.create_image_section()
        content_splitter.addWidget(image_section)
        
        # Right: Form
        form_section = self.create_form_section()
        content_splitter.addWidget(form_section)
        
        # Cân bằng tỷ lệ: ảnh chiếm 60%, form chiếm 40%
        content_splitter.setSizes([1000, 700])
        layout.addWidget(content_splitter, 3)
        
        main_content.setLayout(layout)
        return main_content
    
    def create_guidelines_section(self):
        """Create guidelines section"""
        guidelines_group = QGroupBox("📸 Hướng dẫn chụp ảnh CCCD chuẩn")
        layout = QHBoxLayout()
        
        guidelines_text = QTextBrowser()
        guidelines_text.setMaximumHeight(130)  # Tăng chiều cao
        guidelines_text.setStyleSheet("QTextBrowser { border: none; background-color: transparent; }")
        guidelines_text.setHtml("""
        <div style='font-family: Arial; font-size: 13px; line-height: 1.4;'>
        <b style='color: #28a745;'>Để đảm bảo OCR chính xác, vui lòng:</b><br><br>
        • <b>Ánh sáng:</b> Chụp trong điều kiện ánh sáng tốt, tránh bóng mờ<br>
        • <b>Góc chụp:</b> Đặt CCCD trên nền trắng, phẳng, chụp thẳng góc<br>
        • <b>Khung hình:</b> Đảm bảo toàn bộ thẻ nằm trong khung hình, không bị cắt<br>
        • <b>Độ phân giải:</b> Tối thiểu 1000x600 pixels<br>
        • <b>Định dạng:</b> Hỗ trợ JPG, PNG, BMP
        </div>
        """)
        
        layout.addWidget(guidelines_text)
        guidelines_group.setLayout(layout)
        
        return guidelines_group
    
    def create_image_section(self):
        """Create image upload section"""
        image_group = QGroupBox("📷 Tải ảnh CCCD")
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # Upload buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        self.upload_front_btn = QPushButton("📤 Tải ảnh mặt trước")
        self.upload_back_btn = QPushButton("📤 Tải ảnh mặt sau")
        self.process_btn = QPushButton("🔍 Xử lý OCR")
        self.process_btn.setEnabled(False)
        
        # Tăng kích thước nút
        for btn in [self.upload_front_btn, self.upload_back_btn, self.process_btn]:
            btn.setMinimumHeight(45)
            btn.setMinimumWidth(150)
        
        self.upload_front_btn.clicked.connect(lambda: self.load_image('front'))
        self.upload_back_btn.clicked.connect(lambda: self.load_image('back'))
        self.process_btn.clicked.connect(self.process_images)
        
        btn_layout.addWidget(self.upload_front_btn)
        btn_layout.addWidget(self.upload_back_btn)
        btn_layout.addWidget(self.process_btn)
        btn_layout.addStretch()
        
        # Image display
        image_display_layout = QHBoxLayout()
        image_display_layout.setSpacing(20)  # Tăng khoảng cách giữa 2 ảnh
        
        self.front_image_label = QLabel('📷 Mặt trước')
        self.front_image_label.setAlignment(Qt.AlignCenter)
        self.front_image_label.setMinimumSize(380, 240)  # Tăng kích thước
        self.front_image_label.setStyleSheet("""
            border: 3px dashed #dee2e6; 
            background-color: #f8f9fa; 
            border-radius: 8px;
            font-size: 14px;
            color: #6c757d;
        """)
        
        self.back_image_label = QLabel('📷 Mặt sau')
        self.back_image_label.setAlignment(Qt.AlignCenter)
        self.back_image_label.setMinimumSize(380, 240)  # Tăng kích thước
        self.back_image_label.setStyleSheet("""
            border: 3px dashed #dee2e6; 
            background-color: #f8f9fa; 
            border-radius: 8px;
            font-size: 14px;
            color: #6c757d;
        """)
        
        image_display_layout.addWidget(self.front_image_label)
        image_display_layout.addWidget(self.back_image_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #dee2e6;
                border-radius: 6px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #28a745;
                border-radius: 4px;
            }
        """)
        
        # Status + vùng hiển thị kết quả OCR ngắn gọn để bớt trống
        status_container = QHBoxLayout()
        status_container.setSpacing(10)

        self.status_label = QLabel('✅ Sẵn sàng xử lý')
        self.status_label.setStyleSheet("color: #28a745; font-weight: bold; font-size: 14px; padding: 10px;")

        self.ocr_summary = QTextBrowser()
        self.ocr_summary.setMinimumHeight(90)
        self.ocr_summary.setStyleSheet("QTextBrowser { border: 1px solid #dee2e6; border-radius: 6px; background: #ffffff; }")
        self.ocr_summary.setPlaceholderText("Kết quả OCR tóm tắt sẽ hiển thị ở đây sau khi xử lý…")
        self.ocr_summary.setOpenExternalLinks(False)

        status_container.addWidget(self.status_label, 1)
        status_container.addWidget(self.ocr_summary, 3)
        
        layout.addLayout(btn_layout)
        layout.addLayout(image_display_layout)
        layout.addWidget(self.progress_bar)
        layout.addLayout(status_container)
        
        image_group.setLayout(layout)
        return image_group
    
    def create_form_section(self):
        """Create form section for CCCD information"""
        form_group = QGroupBox("📋 Thông tin CCCD/CMND")
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # Form layout với spacing tốt hơn
        form_layout = QFormLayout()
        form_layout.setVerticalSpacing(12)  # Giảm khoảng cách dọc giữa các dòng
        form_layout.setHorizontalSpacing(15)  # Tăng khoảng cách ngang giữa label và input
        
        # Personal information fields
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Họ và tên đầy đủ")
        form_layout.addRow("Họ và tên:", self.name_input)
        
        # Gender radio buttons - cân bằng 2 cột và canh giữa tuyệt đối
        gender_grid = QGridLayout()
        gender_grid.setContentsMargins(0, 0, 0, 0)
        gender_grid.setHorizontalSpacing(0)
        self.gender_group = QButtonGroup()
        self.male_radio = QRadioButton("Nam")
        self.female_radio = QRadioButton("Nữ")
        self.gender_group.addButton(self.male_radio)
        self.gender_group.addButton(self.female_radio)
        gender_grid.addWidget(self.male_radio, 0, 0, alignment=Qt.AlignCenter)
        gender_grid.addWidget(self.female_radio, 0, 1, alignment=Qt.AlignCenter)
        gender_grid.setColumnStretch(0, 1)
        gender_grid.setColumnStretch(1, 1)
        gender_widget = QWidget()
        gender_widget.setLayout(gender_grid)
        gender_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        form_layout.addRow("Giới tính:", gender_widget)
        
        self.cccd_input = QLineEdit()
        self.cccd_input.setPlaceholderText("Số CCCD/CMND (12 số)")
        form_layout.addRow("Số CCCD/CMND:", self.cccd_input)
        
        self.dob_input = QDateEdit()
        self.dob_input.setDate(QDate.currentDate())
        self.dob_input.setDisplayFormat("dd/MM/yyyy")
        self.dob_input.setCalendarPopup(True)
        form_layout.addRow("Ngày sinh:", self.dob_input)
        
        self.issue_date_input = QDateEdit()
        self.issue_date_input.setDate(QDate.currentDate())
        self.issue_date_input.setDisplayFormat("dd/MM/yyyy")
        self.issue_date_input.setCalendarPopup(True)
        form_layout.addRow("Ngày cấp:", self.issue_date_input)
        
        self.expire_date_input = QDateEdit()
        self.expire_date_input.setDate(QDate.currentDate().addYears(15))
        self.expire_date_input.setDisplayFormat("dd/MM/yyyy")
        self.expire_date_input.setCalendarPopup(True)
        form_layout.addRow("Ngày hết hạn:", self.expire_date_input)
        
        self.origin_place_input = QLineEdit()
        self.origin_place_input.setPlaceholderText("Quê quán")
        form_layout.addRow("Quê quán:", self.origin_place_input)
        
        self.ethnicity_combo = QComboBox()
        self.ethnicity_combo.addItems(self.ethnic_groups)
        form_layout.addRow("Dân tộc:", self.ethnicity_combo)
        
        self.nationality_input = QLineEdit()
        self.nationality_input.setText("Việt Nam")
        form_layout.addRow("Quốc tịch:", self.nationality_input)
        
        self.current_place_input = QTextEdit()
        self.current_place_input.setPlaceholderText("Nơi thường trú")
        self.current_place_input.setMaximumHeight(80)
        form_layout.addRow("Nơi thường trú:", self.current_place_input)
        
        # Buttons với spacing tốt hơn
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        self.save_btn = QPushButton("💾 Lưu thông tin")
        self.clear_form_btn = QPushButton("🗑️ Xóa form")
        
        # Tăng kích thước nút
        for btn in [self.save_btn, self.clear_form_btn]:
            btn.setMinimumHeight(45)
            btn.setMinimumWidth(140)
        
        self.save_btn.clicked.connect(self.save_record)
        self.clear_form_btn.clicked.connect(self.clear_form)
        
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.clear_form_btn)
        button_layout.addStretch()
        
        layout.addLayout(form_layout)
        layout.addLayout(button_layout)
        
        form_group.setLayout(layout)
        return form_group
    
    def menu_item_clicked(self):
        """Handle menu item clicks"""
        sender = self.sender()
        if sender.text() == "CMND/CCCD":
            # Already active
            pass
    
    def show_coming_soon(self, feature_name):
        """Show coming soon message"""
        QMessageBox.information(self, "Thông báo", f"Tính năng {feature_name} đang được phát triển!")
    
    def load_image(self, image_type):
        """Load image for front or back"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Chọn ảnh {image_type}", "", 
            "Image files (*.jpg *.jpeg *.png *.bmp);;All files (*)"
        )
        
        if file_path:
            if image_type == 'front':
                self.front_image_path = file_path
                self.upload_front_btn.setText(f"✓ {os.path.basename(file_path)}")
                pixmap = QPixmap(file_path)
                scaled_pixmap = pixmap.scaled(380, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.front_image_label.setPixmap(scaled_pixmap)
            else:
                self.back_image_path = file_path
                self.upload_back_btn.setText(f"✓ {os.path.basename(file_path)}")
                pixmap = QPixmap(file_path)
                scaled_pixmap = pixmap.scaled(380, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.back_image_label.setPixmap(scaled_pixmap)
            
            # Enable process button if at least one image is loaded
            if self.front_image_path or self.back_image_path:
                self.process_btn.setEnabled(True)
    
    def process_images(self):
        """Process images with OCR"""
        if not self.front_image_path and not self.back_image_path:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng tải ít nhất 1 ảnh!")
            return
        
        # Check model paths
        yolo_model_path = os.path.join(SCRIPT_DIR, "Model", "best.pt")
        ocr_model_path = MODEL_DIR
        
        if not os.path.exists(yolo_model_path):
            QMessageBox.critical(self, "Lỗi", f"Không tìm thấy YOLO model: {yolo_model_path}")
            return
        
        if not os.path.exists(ocr_model_path):
            QMessageBox.critical(self, "Lỗi", f"Không tìm thấy OCR model: {ocr_model_path}")
            return
        
        # Disable buttons and show progress
        self.set_processing_mode(True)
        
        # Start detection
        image_paths = [self.front_image_path, self.back_image_path]
        self.detection_thread = DetectionThread(image_paths, yolo_model_path)
        self.detection_thread.progress.connect(self.update_progress)
        self.detection_thread.finished.connect(self.on_detection_finished)
        self.detection_thread.error.connect(self.on_error)
        self.detection_thread.start()
    
    def set_processing_mode(self, processing):
        """Enable/disable UI during processing"""
        self.upload_front_btn.setEnabled(not processing)
        self.upload_back_btn.setEnabled(not processing)
        self.process_btn.setEnabled(not processing)
        self.progress_bar.setVisible(processing)
        
        if processing:
            self.status_label.setText("Đang xử lý...")
            self.status_label.setStyleSheet("color: blue; font-weight: bold;")
        else:
            self.status_label.setText("Sẵn sàng xử lý")
            self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def on_detection_finished(self, detection_results):
        """Handle detection completion"""
        self.all_detection_results = detection_results
        self.status_label.setText("Đang thực hiện OCR...")
        
        # Start OCR
        self.ocr_thread = OCRThread(detection_results)
        self.ocr_thread.progress.connect(self.update_progress)
        self.ocr_thread.finished.connect(self.on_ocr_finished)
        self.ocr_thread.error.connect(self.on_error)
        self.ocr_thread.start()
    
    def on_ocr_finished(self, ocr_results):
        """Handle OCR completion and fill form"""
        self.current_results = ocr_results
        self.fill_form_with_ocr_results(ocr_results)
        self.set_processing_mode(False)
        self.status_label.setText("OCR hoàn thành!")
        self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        self.update_ocr_summary(ocr_results)
    
    def on_error(self, error_msg):
        """Handle processing errors"""
        QMessageBox.critical(self, "Lỗi", f"Đã xảy ra lỗi: {error_msg}")
        self.set_processing_mode(False)
        self.status_label.setText(f"Lỗi: {error_msg}")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        if hasattr(self, 'ocr_summary'):
            self.ocr_summary.setHtml(f"<span style='color:#dc3545;font-weight:bold;'>Lỗi:</span> {error_msg}")
    
    def fill_form_with_ocr_results(self, ocr_results):
        """Fill form fields with OCR results"""
        # Name
        if 'name' in ocr_results and ocr_results['name']:
            self.name_input.setText(ocr_results['name'])
        
        # Gender
        if 'gender' in ocr_results and ocr_results['gender']:
            gender = ocr_results['gender'].lower()
            if 'nam' in gender:
                self.male_radio.setChecked(True)
            elif 'nữ' in gender or 'nu' in gender:
                self.female_radio.setChecked(True)
        
        # CCCD number
        if 'id' in ocr_results and ocr_results['id']:
            self.cccd_input.setText(ocr_results['id'])
        
        # Dates
        if 'dob' in ocr_results and ocr_results['dob']:
            date_str = self.parse_date_string(ocr_results['dob'])
            if date_str:
                self.dob_input.setDate(QDate.fromString(date_str, "dd/MM/yyyy"))
        
        if 'issue_date' in ocr_results and ocr_results['issue_date']:
            date_str = self.parse_date_string(ocr_results['issue_date'])
            if date_str:
                self.issue_date_input.setDate(QDate.fromString(date_str, "dd/MM/yyyy"))
        
        if 'expire_date' in ocr_results and ocr_results['expire_date']:
            date_str = self.parse_date_string(ocr_results['expire_date'])
            if date_str:
                self.expire_date_input.setDate(QDate.fromString(date_str, "dd/MM/yyyy"))
        
        # Places
        if 'origin_place' in ocr_results and ocr_results['origin_place']:
            self.origin_place_input.setText(ocr_results['origin_place'])
        
        if 'current_place' in ocr_results and ocr_results['current_place']:
            self.current_place_input.setPlainText(ocr_results['current_place'])
        
        # Nationality
        if 'nationality' in ocr_results and ocr_results['nationality']:
            self.nationality_input.setText(ocr_results['nationality'])

    def update_ocr_summary(self, ocr_results):
        """Render a compact HTML summary of OCR results"""
        if not hasattr(self, 'ocr_summary'):
            return
        if not ocr_results:
            self.ocr_summary.clear()
            return
        def esc(s):
            return (s or '').replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
        rows = []
        mapping = [
            ("Họ tên", esc(ocr_results.get('name', ''))),
            ("Số CCCD", esc(ocr_results.get('id', ''))),
            ("Giới tính", esc(ocr_results.get('gender', ''))),
            ("Ngày sinh", esc(ocr_results.get('dob', ''))),
            ("Ngày cấp", esc(ocr_results.get('issue_date', ''))),
            ("Hết hạn", esc(ocr_results.get('expire_date', ''))),
            ("Quê quán", esc(ocr_results.get('origin_place', ''))),
            ("Thường trú", esc(ocr_results.get('current_place', ''))),
            ("Quốc tịch", esc(ocr_results.get('nationality', ''))),
        ]
        for k, v in mapping:
            if v:
                rows.append(f"<tr><td style='padding:4px 8px;color:#495057;white-space:nowrap;'>{k}</td><td style='padding:4px 8px;color:#212529;'><b>{v}</b></td></tr>")
        html = """
        <div style='font-family:Arial; font-size:12px;'>
          <div style='margin-bottom:6px;color:#0d6efd;font-weight:bold;'>Tóm tắt kết quả OCR</div>
          <table style='border-collapse:collapse; width:100%;'>
            {rows}
          </table>
        </div>
        """.replace('{rows}', '\n'.join(rows))
        self.ocr_summary.setHtml(html)
    
    def parse_date_string(self, date_str):
        """Parse various date formats to dd/MM/yyyy"""
        import re
        
        # Remove extra spaces and common prefixes
        date_str = re.sub(r'\s+', ' ', date_str.strip())
        
        # Try different date patterns
        patterns = [
            r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})',  # dd/mm/yyyy or dd-mm-yyyy
            r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2})',   # dd/mm/yy
            r'(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})',   # yyyy/mm/dd
        ]
        
        for pattern in patterns:
            match = re.search(pattern, date_str)
            if match:
                groups = match.groups()
                if len(groups[2]) == 2:  # 2-digit year
                    year = int(groups[2])
                    if year > 50:
                        year += 1900
                    else:
                        year += 2000
                    groups = (groups[0], groups[1], str(year))
                
                try:
                    if len(groups[0]) == 4:  # yyyy format first
                        day, month, year = groups[2], groups[1], groups[0]
                    else:
                        day, month, year = groups[0], groups[1], groups[2]
                    
                    # Validate date
                    datetime.strptime(f"{day}/{month}/{year}", "%d/%m/%Y")
                    return f"{day.zfill(2)}/{month.zfill(2)}/{year}"
                except ValueError:
                    continue
        
        return None
    
    def save_record(self):
        """Save current form data to database"""
        # Validate required fields
        if not self.name_input.text().strip():
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập họ tên!")
            return
        
        if not self.cccd_input.text().strip():
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập số CCCD/CMND!")
            return
        
        # Prepare data
        data = {
            'ho_ten': self.name_input.text().strip(),
            'gioi_tinh': 'Nam' if self.male_radio.isChecked() else 'Nữ' if self.female_radio.isChecked() else '',
            'so_cccd': self.cccd_input.text().strip(),
            'ngay_sinh': self.dob_input.date().toString("yyyy-MM-dd"),
            'ngay_cap': self.issue_date_input.date().toString("yyyy-MM-dd"),
            'ngay_het_han': self.expire_date_input.date().toString("yyyy-MM-dd"),
            'que_quan': self.origin_place_input.text().strip(),
            'dan_toc': self.ethnicity_combo.currentText(),
            'quoc_tich': self.nationality_input.text().strip(),
            'noi_thuong_tru': self.current_place_input.toPlainText().strip(),
            'image_front_path': self.front_image_path or '',
            'image_back_path': self.back_image_path or ''
        }
        
        # Save to database
        if self.db_manager.save_record(data):
            QMessageBox.information(self, "Thành công", "Đã lưu thông tin thành công!")
            self.load_recent_records()
        else:
            QMessageBox.critical(self, "Lỗi", "Không thể lưu thông tin!")
    
    def clear_form(self):
        """Clear all form fields"""
        self.name_input.clear()
        self.male_radio.setChecked(False)
        self.female_radio.setChecked(False)
        self.cccd_input.clear()
        self.dob_input.setDate(QDate.currentDate())
        self.issue_date_input.setDate(QDate.currentDate())
        self.expire_date_input.setDate(QDate.currentDate().addYears(15))
        self.origin_place_input.clear()
        self.ethnicity_combo.setCurrentIndex(0)
        self.nationality_input.setText("Việt Nam")
        self.current_place_input.clear()
        
        # Clear images
        self.front_image_path = None
        self.back_image_path = None
        self.front_image_label.clear()
        self.front_image_label.setText('Mặt trước')
        self.back_image_label.clear()
        self.back_image_label.setText('Mặt sau')
        self.upload_front_btn.setText("Tải ảnh mặt trước")
        self.upload_back_btn.setText("Tải ảnh mặt sau")
        self.process_btn.setEnabled(False)
    
    def search_record(self):
        """Search record by CCCD number"""
        cccd_number = self.search_input.text().strip()
        if not cccd_number:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập số CCCD để tìm kiếm!")
            return
        
        record = self.db_manager.search_by_cccd(cccd_number)
        if record:
            self.load_record_to_form(record)
            QMessageBox.information(self, "Tìm thấy", f"Đã tải thông tin của {record['ho_ten']}")
        else:
            QMessageBox.information(self, "Không tìm thấy", "Không tìm thấy CCCD này trong cơ sở dữ liệu!")
    
    def load_record_to_form(self, record):
        """Load database record to form"""
        self.name_input.setText(record.get('ho_ten', ''))
        
        gender = record.get('gioi_tinh', '')
        if gender == 'Nam':
            self.male_radio.setChecked(True)
        elif gender == 'Nữ':
            self.female_radio.setChecked(True)
        
        self.cccd_input.setText(record.get('so_cccd', ''))
        
        # Dates
        if record.get('ngay_sinh'):
            try:
                date_obj = datetime.strptime(record['ngay_sinh'], '%Y-%m-%d').date()
                self.dob_input.setDate(QDate(date_obj))
            except:
                pass
        
        if record.get('ngay_cap'):
            try:
                date_obj = datetime.strptime(record['ngay_cap'], '%Y-%m-%d').date()
                self.issue_date_input.setDate(QDate(date_obj))
            except:
                pass
        
        if record.get('ngay_het_han'):
            try:
                date_obj = datetime.strptime(record['ngay_het_han'], '%Y-%m-%d').date()
                self.expire_date_input.setDate(QDate(date_obj))
            except:
                pass
        
        self.origin_place_input.setText(record.get('que_quan', ''))
        
        ethnicity = record.get('dan_toc', '')
        if ethnicity in self.ethnic_groups:
            self.ethnicity_combo.setCurrentText(ethnicity)
        
        self.nationality_input.setText(record.get('quoc_tich', ''))
        self.current_place_input.setPlainText(record.get('noi_thuong_tru', ''))
        
        # Load images if paths exist
        if record.get('image_front_path') and os.path.exists(record['image_front_path']):
            self.front_image_path = record['image_front_path']
            pixmap = QPixmap(self.front_image_path)
            scaled_pixmap = pixmap.scaled(380, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.front_image_label.setPixmap(scaled_pixmap)
            self.upload_front_btn.setText(f"✓ {os.path.basename(self.front_image_path)}")
        
        if record.get('image_back_path') and os.path.exists(record['image_back_path']):
            self.back_image_path = record['image_back_path']
            pixmap = QPixmap(self.back_image_path)
            scaled_pixmap = pixmap.scaled(380, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.back_image_label.setPixmap(scaled_pixmap)
            self.upload_back_btn.setText(f"✓ {os.path.basename(self.back_image_path)}")
    
    def load_recent_records(self):
        """Load recent records to sidebar list"""
        self.recent_list.clear()
        records = self.db_manager.get_all_records()
        
        for cccd, name, dob in records[:10]:  # Show last 10 records
            item_text = f"{name}\n{cccd}"
            if dob:
                try:
                    date_obj = datetime.strptime(dob, '%Y-%m-%d')
                    item_text += f"\n{date_obj.strftime('%d/%m/%Y')}"
                except:
                    pass

            # Lấy bản ghi đầy đủ để tạo thumbnail ảnh trước/sau
            record = self.db_manager.search_by_cccd(cccd)
            icon_pixmap = QPixmap(128, 48)
            icon_pixmap.fill(Qt.transparent)

            painter = QPainter(icon_pixmap)
            try:
                # Ảnh mặt trước
                if record and record.get('image_front_path') and os.path.exists(record['image_front_path']):
                    front_pm = QPixmap(record['image_front_path']).scaled(64, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                else:
                    front_pm = QPixmap(64, 48)
                    front_pm.fill(QColor('#e9ecef'))
                painter.drawPixmap(0, 0, 64, 48, front_pm)

                # Ảnh mặt sau
                if record and record.get('image_back_path') and os.path.exists(record['image_back_path']):
                    back_pm = QPixmap(record['image_back_path']).scaled(64, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                else:
                    back_pm = QPixmap(64, 48)
                    back_pm.fill(QColor('#e9ecef'))
                painter.drawPixmap(64, 0, 64, 48, back_pm)
            finally:
                painter.end()

            item = QListWidgetItem(QIcon(icon_pixmap), item_text)
            item.setSizeHint(QSize(220, 72))
            item.setData(Qt.UserRole, cccd)  # Store CCCD for lookup
            self.recent_list.addItem(item)
    
    def load_recent_record(self, item):
        """Load selected recent record"""
        cccd = item.data(Qt.UserRole)
        record = self.db_manager.search_by_cccd(cccd)
        if record:
            self.load_record_to_form(record)

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    app.setApplicationName("Document Management System")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("CCCD Processing Tools")
    
    window = EnhancedCCCDApp()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()