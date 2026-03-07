import os
import sys
import json
import uuid
from datetime import datetime
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QTabWidget,
    QSlider, QGroupBox, QStatusBar, QFrame,
    QScrollArea, QSizePolicy, QTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QProgressBar, QDialog, QLineEdit,
    QGridLayout, QSpacerItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPainter, QLinearGradient

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import UNet, CLASS_NAMES, CLASS_COLORS


CLASS_NAMES_EN = {
    0: "Other",
    1: "Urban Areas",
    2: "Water Bodies",
    3: "Vegetation"
}

COLORS = {
    'bg_dark': '#0f0f1a',
    'bg_card': '#1a1a2e',
    'bg_hover': '#252542',
    'accent_blue': '#00d4ff',
    'accent_purple': '#bd00ff',
    'accent_green': '#00ff9d',
    'accent_orange': '#ff6b35',
    'text_primary': '#ffffff',
    'text_secondary': '#8892b0',
    'border': '#2d2d4a',
}


class PredictionThread(QThread):
    """Background thread for model inference"""
    finished = pyqtSignal(np.ndarray, np.ndarray)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, model, image, device, image_size=256):
        super().__init__()
        self.model = model
        self.image = image
        self.device = device
        self.image_size = image_size
    
    def run(self):
        try:
            self.progress.emit(20)
            original_size = self.image.shape[:2]
            
            transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            
            transformed = transform(image=self.image)
            image_tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            self.progress.emit(50)
            
            self.model.eval()
            with torch.no_grad():
                output = self.model(image_tensor)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1)
            
            self.progress.emit(80)
            
            pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)
            probs_np = probs.squeeze().cpu().numpy()
            
            pred_resized = np.array(Image.fromarray(pred_np).resize(
                (original_size[1], original_size[0]), Image.NEAREST))
            
            probs_resized = np.zeros((4, original_size[0], original_size[1]))
            for i in range(4):
                probs_resized[i] = np.array(Image.fromarray(probs_np[i]).resize(
                    (original_size[1], original_size[0]), Image.BILINEAR))
            
            self.progress.emit(100)
            self.finished.emit(pred_resized, probs_resized)
            
        except Exception as e:
            self.error.emit(str(e))


class ImageLabel(QLabel):
    """Scalable image display widget"""
    
    def __init__(self, placeholder_text=""):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.placeholder_text = placeholder_text
        self._pixmap = None
        self.update_style()
    
    def update_style(self):
        self.setStyleSheet(f"""
            QLabel {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {COLORS['bg_card']}, stop:1 {COLORS['bg_dark']});
                border: 2px dashed {COLORS['border']};
                border-radius: 16px;
                color: {COLORS['text_secondary']};
                font-size: 14px;
            }}
        """)
        if not self._pixmap:
            self.setText(self.placeholder_text)
    
    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self._update_scaled_pixmap()
    
    def _update_scaled_pixmap(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size() - QSize(20, 20),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            super().setPixmap(scaled)
    
    def resizeEvent(self, event):
        self._update_scaled_pixmap()
        super().resizeEvent(event)
    
    def clear_image(self):
        self._pixmap = None
        super().setPixmap(QPixmap())
        self.setText(self.placeholder_text)


class ModernButton(QPushButton):
    """Modern styled button"""
    
    def __init__(self, text, color=None, icon_text=""):
        super().__init__(f"{icon_text} {text}" if icon_text else text)
        self.color = color or COLORS['accent_blue']
        self.update_style()
    
    def update_style(self):
        self.setStyleSheet(f"""
            QPushButton {{
                background: {self.color};
                color: #0f0f1a;
                border: none;
                border-radius: 12px;
                padding: 14px 28px;
                font-weight: bold;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background: {self.color}dd;
            }}
            QPushButton:pressed {{
                background: {self.color}aa;
            }}
            QPushButton:disabled {{
                background: {COLORS['border']};
                color: {COLORS['text_secondary']};
            }}
        """)


class StatsWidget(QFrame):
    """Statistics display widget with visual bars"""
    
    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['bg_card']};
                border-radius: 16px;
                padding: 10px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        title = QLabel("Segmentation Statistics")
        title.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        self.bars = {}
        self.labels = {}
        
        class_colors = {
            0: COLORS['text_secondary'],
            1: '#ff4757',
            2: '#3742fa',
            3: '#2ed573'
        }
        
        for class_id, name in CLASS_NAMES_EN.items():
            row = QHBoxLayout()
            
            name_label = QLabel(name)
            name_label.setStyleSheet(f"color: {class_colors[class_id]}; font-weight: bold; min-width: 120px;")
            row.addWidget(name_label)
            
            bar_container = QFrame()
            bar_container.setFixedHeight(24)
            bar_container.setStyleSheet(f"background: {COLORS['bg_dark']}; border-radius: 12px;")
            bar_layout = QHBoxLayout(bar_container)
            bar_layout.setContentsMargins(0, 0, 0, 0)
            
            bar = QFrame()
            bar.setFixedHeight(24)
            bar.setStyleSheet(f"background: {class_colors[class_id]}; border-radius: 12px;")
            bar.setFixedWidth(0)
            bar_layout.addWidget(bar)
            bar_layout.addStretch()
            
            self.bars[class_id] = bar
            row.addWidget(bar_container, 1)
            
            pct_label = QLabel("0.0%")
            pct_label.setStyleSheet(f"color: {COLORS['text_primary']}; min-width: 60px; text-align: right;")
            pct_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.labels[class_id] = pct_label
            row.addWidget(pct_label)
            
            layout.addLayout(row)
    
    def update_stats(self, mask):
        total = mask.size
        max_width = 200
        
        for class_id in range(4):
            count = np.sum(mask == class_id)
            pct = (count / total) * 100
            self.labels[class_id].setText(f"{pct:.1f}%")
            self.bars[class_id].setFixedWidth(int(max_width * pct / 100))


class LegendWidget(QFrame):
    """Color legend widget"""
    
    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['bg_card']};
                border-radius: 16px;
                padding: 15px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        
        title = QLabel("Legend")
        title.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        class_colors_rgb = {
            0: (128, 128, 128),
            1: (255, 71, 87),
            2: (55, 66, 250),
            3: (46, 213, 115)
        }
        
        for class_id, name in CLASS_NAMES_EN.items():
            color = class_colors_rgb[class_id]
            item = QLabel(name)
            item.setStyleSheet(f"color: rgb{color}; font-weight: bold; font-size: 13px;")
            layout.addWidget(item)


class AddComparisonDialog(QDialog):
    """Dialog for adding new comparison"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Comparison")
        self.setMinimumSize(600, 500)
        self.setStyleSheet(f"""
            QDialog {{
                background: {COLORS['bg_dark']};
            }}
            QLabel {{
                color: {COLORS['text_primary']};
            }}
            QLineEdit, QTextEdit {{
                background: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                padding: 10px;
                font-size: 13px;
            }}
            QLineEdit:focus, QTextEdit:focus {{
                border-color: {COLORS['accent_blue']};
            }}
        """)
        
        self.model_image_path = None
        self.ground_truth_path = None
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Title
        title = QLabel("Add New Comparison")
        title.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {COLORS['accent_blue']};")
        layout.addWidget(title)
        
        # Title input
        layout.addWidget(QLabel("Title:"))
        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("e.g., Analysis of Amazon Rainforest Region")
        layout.addWidget(self.title_input)
        
        # Images row
        images_layout = QHBoxLayout()
        
        # Model result
        model_group = QVBoxLayout()
        model_group.addWidget(QLabel("Model Result:"))
        self.model_preview = ImageLabel("Click to select...")
        self.model_preview.setFixedSize(200, 150)
        self.model_preview.mousePressEvent = lambda e: self.select_model_image()
        model_group.addWidget(self.model_preview)
        images_layout.addLayout(model_group)
        
        # Ground truth
        gt_group = QVBoxLayout()
        gt_group.addWidget(QLabel("Ground Truth:"))
        self.gt_preview = ImageLabel("Click to select...")
        self.gt_preview.setFixedSize(200, 150)
        self.gt_preview.mousePressEvent = lambda e: self.select_gt_image()
        gt_group.addWidget(self.gt_preview)
        images_layout.addLayout(gt_group)
        
        layout.addLayout(images_layout)
        
        # Description
        layout.addWidget(QLabel("Description / Analysis:"))
        self.description_input = QTextEdit()
        self.description_input.setPlaceholderText(
            "Write your analysis here...\n\n"
            "Example:\n"
            "- Model correctly identified 85% of urban areas\n"
            "- Water bodies detection accuracy: 92%\n"
            "- Some confusion between agriculture and vegetation"
        )
        self.description_input.setMinimumHeight(150)
        layout.addWidget(self.description_input)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        cancel_btn = ModernButton("Cancel", COLORS['text_secondary'])
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        save_btn = ModernButton("Save Comparison", COLORS['accent_green'])
        save_btn.clicked.connect(self.accept)
        btn_layout.addWidget(save_btn)
        
        layout.addLayout(btn_layout)
    
    def select_model_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model Result Image", "", 
                                               "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.model_image_path = path
            pixmap = QPixmap(path)
            self.model_preview.setPixmap(pixmap)
    
    def select_gt_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Ground Truth Image", "",
                                               "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.ground_truth_path = path
            pixmap = QPixmap(path)
            self.gt_preview.setPixmap(pixmap)
    
    def get_data(self):
        return {
            'title': self.title_input.text(),
            'description': self.description_input.toPlainText(),
            'model_image': self.model_image_path,
            'ground_truth': self.ground_truth_path
        }


class ComparisonsTab(QWidget):
    """Tab 1: Comparisons Gallery"""
    
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.comparisons = []
        self.current_index = 0
        
        self.load_comparisons()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Header
        header = QHBoxLayout()
        
        title = QLabel("Comparisons Gallery")
        title.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {COLORS['text_primary']};")
        header.addWidget(title)
        
        header.addStretch()
        
        add_btn = ModernButton("Add Comparison", COLORS['accent_purple'])
        add_btn.clicked.connect(self.add_comparison)
        header.addWidget(add_btn)
        
        layout.addLayout(header)
        
        # Navigation
        nav_layout = QHBoxLayout()
        
        self.prev_btn = ModernButton("Previous", COLORS['text_secondary'])
        self.prev_btn.clicked.connect(self.show_previous)
        nav_layout.addWidget(self.prev_btn)
        
        nav_layout.addStretch()
        
        self.page_label = QLabel("0 / 0")
        self.page_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 18px; font-weight: bold;")
        nav_layout.addWidget(self.page_label)
        
        nav_layout.addStretch()
        
        self.next_btn = ModernButton("Next", COLORS['text_secondary'])
        self.next_btn.clicked.connect(self.show_next)
        nav_layout.addWidget(self.next_btn)
        
        layout.addLayout(nav_layout)
        
        # Content area
        content = QFrame()
        content.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['bg_card']};
                border-radius: 20px;
            }}
        """)
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(20)
        content_layout.setContentsMargins(30, 30, 30, 30)
        
        # Title
        self.comparison_title = QLabel("No comparisons yet")
        self.comparison_title.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {COLORS['accent_blue']};")
        self.comparison_title.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(self.comparison_title)
        
        # Images
        images_layout = QHBoxLayout()
        
        # Model result
        model_container = QVBoxLayout()
        model_label = QLabel("Model Result")
        model_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 14px;")
        model_label.setAlignment(Qt.AlignCenter)
        model_container.addWidget(model_label)
        
        self.model_image = ImageLabel("Model prediction will appear here")
        self.model_image.setMinimumSize(400, 350)
        model_container.addWidget(self.model_image)
        images_layout.addLayout(model_container)
        
        # VS label
        vs_label = QLabel("VS")
        vs_label.setStyleSheet(f"color: {COLORS['accent_orange']}; font-size: 24px; font-weight: bold;")
        vs_label.setAlignment(Qt.AlignCenter)
        images_layout.addWidget(vs_label)
        
        # Ground truth
        gt_container = QVBoxLayout()
        gt_label = QLabel("Ground Truth")
        gt_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 14px;")
        gt_label.setAlignment(Qt.AlignCenter)
        gt_container.addWidget(gt_label)
        
        self.gt_image = ImageLabel("Ground truth will appear here")
        self.gt_image.setMinimumSize(400, 350)
        gt_container.addWidget(self.gt_image)
        images_layout.addLayout(gt_container)
        
        content_layout.addLayout(images_layout)
        
        # Description
        desc_label = QLabel("Analysis")
        desc_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 14px;")
        content_layout.addWidget(desc_label)
        
        self.description_text = QLabel("Add your first comparison to see analysis here.")
        self.description_text.setStyleSheet(f"""
            color: {COLORS['text_primary']}; 
            font-size: 14px; 
            line-height: 1.6;
            background: {COLORS['bg_dark']};
            padding: 20px;
            border-radius: 12px;
        """)
        self.description_text.setWordWrap(True)
        self.description_text.setMinimumHeight(100)
        content_layout.addWidget(self.description_text)
        
        # Delete button
        delete_layout = QHBoxLayout()
        delete_layout.addStretch()
        self.delete_btn = ModernButton("Delete", "#ff4757")
        self.delete_btn.clicked.connect(self.delete_current)
        self.delete_btn.setVisible(False)
        delete_layout.addWidget(self.delete_btn)
        content_layout.addLayout(delete_layout)
        
        layout.addWidget(content)
        
        self.update_display()
    
    def load_comparisons(self):
        comparisons_file = os.path.join(self.data_dir, 'comparisons.json')
        if os.path.exists(comparisons_file):
            with open(comparisons_file, 'r', encoding='utf-8') as f:
                self.comparisons = json.load(f)
    
    def save_comparisons(self):
        os.makedirs(self.data_dir, exist_ok=True)
        comparisons_file = os.path.join(self.data_dir, 'comparisons.json')
        with open(comparisons_file, 'w', encoding='utf-8') as f:
            json.dump(self.comparisons, f, indent=2, ensure_ascii=False)
    
    def add_comparison(self):
        dialog = AddComparisonDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            data = dialog.get_data()
            
            if not data['title'] or not data['model_image'] or not data['ground_truth']:
                QMessageBox.warning(self, "Missing Data", "Please fill in title and select both images.")
                return
            
            # Copy images to data directory
            images_dir = os.path.join(self.data_dir, 'comparison_images')
            os.makedirs(images_dir, exist_ok=True)
            
            comp_id = str(uuid.uuid4())[:8]
            
            model_ext = os.path.splitext(data['model_image'])[1]
            gt_ext = os.path.splitext(data['ground_truth'])[1]
            
            model_dest = os.path.join(images_dir, f"{comp_id}_model{model_ext}")
            gt_dest = os.path.join(images_dir, f"{comp_id}_gt{gt_ext}")
            
            import shutil
            shutil.copy(data['model_image'], model_dest)
            shutil.copy(data['ground_truth'], gt_dest)
            
            comparison = {
                'id': comp_id,
                'title': data['title'],
                'description': data['description'],
                'model_image': model_dest,
                'ground_truth': gt_dest,
                'created': datetime.now().isoformat()
            }
            
            self.comparisons.append(comparison)
            self.save_comparisons()
            self.current_index = len(self.comparisons) - 1
            self.update_display()
    
    def delete_current(self):
        if not self.comparisons:
            return
        
        reply = QMessageBox.question(self, "Delete Comparison", 
                                     "Are you sure you want to delete this comparison?",
                                     QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            comp = self.comparisons[self.current_index]
            
            # Delete images
            if os.path.exists(comp.get('model_image', '')):
                os.remove(comp['model_image'])
            if os.path.exists(comp.get('ground_truth', '')):
                os.remove(comp['ground_truth'])
            
            del self.comparisons[self.current_index]
            self.save_comparisons()
            
            if self.current_index >= len(self.comparisons):
                self.current_index = max(0, len(self.comparisons) - 1)
            
            self.update_display()
    
    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
    
    def show_next(self):
        if self.current_index < len(self.comparisons) - 1:
            self.current_index += 1
            self.update_display()
    
    def update_display(self):
        n = len(self.comparisons)
        
        self.prev_btn.setEnabled(self.current_index > 0)
        self.next_btn.setEnabled(self.current_index < n - 1)
        self.delete_btn.setVisible(n > 0)
        
        if n == 0:
            self.page_label.setText("0 / 0")
            self.comparison_title.setText("No comparisons yet")
            self.description_text.setText("Click 'Add Comparison' to create your first analysis comparison.")
            self.model_image.clear_image()
            self.gt_image.clear_image()
            return
        
        self.page_label.setText(f"{self.current_index + 1} / {n}")
        
        comp = self.comparisons[self.current_index]
        self.comparison_title.setText(comp.get('title', 'Untitled'))
        self.description_text.setText(comp.get('description', 'No description'))
        
        if os.path.exists(comp.get('model_image', '')):
            self.model_image.setPixmap(QPixmap(comp['model_image']))
        else:
            self.model_image.clear_image()
        
        if os.path.exists(comp.get('ground_truth', '')):
            self.gt_image.setPixmap(QPixmap(comp['ground_truth']))
        else:
            self.gt_image.clear_image()


class AnalysisTab(QWidget):
    """Tab 2: New Analysis"""
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.current_image = None
        self.current_mask = None
        self.current_overlay = None
        
        self.init_ui()
    
    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Left panel - controls
        left_panel = QFrame()
        left_panel.setMaximumWidth(350)
        left_panel.setStyleSheet(f"background: {COLORS['bg_card']}; border-radius: 20px;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(20)
        left_layout.setContentsMargins(25, 25, 25, 25)
        
        # Title
        title = QLabel("New Analysis")
        title.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {COLORS['text_primary']};")
        left_layout.addWidget(title)
        
        # Model status
        model_frame = QFrame()
        model_frame.setStyleSheet(f"background: {COLORS['bg_dark']}; border-radius: 12px; padding: 15px;")
        model_layout = QVBoxLayout(model_frame)
        
        model_title = QLabel("Model")
        model_title.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        model_layout.addWidget(model_title)
        
        self.model_status = QLabel("Not loaded")
        self.model_status.setStyleSheet(f"color: #ff4757; font-weight: bold;")
        model_layout.addWidget(self.model_status)
        
        model_btn_layout = QHBoxLayout()
        load_model_btn = ModernButton("Load", COLORS['accent_blue'])
        load_model_btn.clicked.connect(self.load_model)
        model_btn_layout.addWidget(load_model_btn)
        
        demo_btn = ModernButton("Demo", COLORS['accent_green'])
        demo_btn.clicked.connect(self.create_demo_model)
        model_btn_layout.addWidget(demo_btn)
        model_layout.addLayout(model_btn_layout)
        
        left_layout.addWidget(model_frame)
        
        # Upload image
        upload_btn = ModernButton("Upload Satellite Image", COLORS['accent_purple'])
        upload_btn.clicked.connect(self.upload_image)
        left_layout.addWidget(upload_btn)
        
        self.image_info = QLabel("No image loaded")
        self.image_info.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.image_info.setWordWrap(True)
        left_layout.addWidget(self.image_info)
        
        # Run analysis
        self.analyze_btn = ModernButton("Run Segmentation", COLORS['accent_orange'])
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.analyze_btn.setEnabled(False)
        left_layout.addWidget(self.analyze_btn)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {COLORS['bg_dark']};
                border-radius: 10px;
                height: 20px;
                text-align: center;
                color: {COLORS['text_primary']};
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['accent_blue']}, stop:1 {COLORS['accent_purple']});
                border-radius: 10px;
            }}
        """)
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        # Transparency slider
        slider_frame = QFrame()
        slider_frame.setStyleSheet(f"background: {COLORS['bg_dark']}; border-radius: 12px; padding: 10px;")
        slider_layout = QVBoxLayout(slider_frame)
        
        slider_label = QLabel("Overlay Transparency")
        slider_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        slider_layout.addWidget(slider_label)
        
        slider_row = QHBoxLayout()
        self.transparency_slider = QSlider(Qt.Horizontal)
        self.transparency_slider.setRange(0, 100)
        self.transparency_slider.setValue(50)
        self.transparency_slider.valueChanged.connect(self.update_overlay)
        self.transparency_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 8px;
                background: {COLORS['border']};
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {COLORS['accent_blue']};
                width: 20px;
                margin: -6px 0;
                border-radius: 10px;
            }}
        """)
        slider_row.addWidget(self.transparency_slider)
        
        self.transparency_label = QLabel("50%")
        self.transparency_label.setStyleSheet(f"color: {COLORS['text_primary']}; min-width: 40px;")
        slider_row.addWidget(self.transparency_label)
        slider_layout.addLayout(slider_row)
        
        left_layout.addWidget(slider_frame)
        
        # Legend
        self.legend = LegendWidget()
        left_layout.addWidget(self.legend)
        
        left_layout.addStretch()
        
        # Save to My Results
        self.save_btn = ModernButton("Save to My Results", COLORS['accent_green'])
        self.save_btn.clicked.connect(self.save_to_results)
        self.save_btn.setEnabled(False)
        left_layout.addWidget(self.save_btn)
        
        layout.addWidget(left_panel)
        
        # Right panel - results
        right_panel = QFrame()
        right_panel.setStyleSheet(f"background: {COLORS['bg_card']}; border-radius: 20px;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(15)
        right_layout.setContentsMargins(25, 25, 25, 25)
        
        # Images grid
        images_layout = QHBoxLayout()
        
        # Original
        orig_container = QVBoxLayout()
        orig_label = QLabel("Original Image")
        orig_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        orig_label.setAlignment(Qt.AlignCenter)
        orig_container.addWidget(orig_label)
        
        self.original_view = ImageLabel("Upload an image to start")
        orig_container.addWidget(self.original_view)
        images_layout.addLayout(orig_container)
        
        # Segmentation
        seg_container = QVBoxLayout()
        seg_label = QLabel("Segmentation Mask")
        seg_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        seg_label.setAlignment(Qt.AlignCenter)
        seg_container.addWidget(seg_label)
        
        self.mask_view = ImageLabel("Run analysis to see results")
        seg_container.addWidget(self.mask_view)
        images_layout.addLayout(seg_container)
        
        right_layout.addLayout(images_layout)
        
        # Overlay
        overlay_label = QLabel("Overlay View")
        overlay_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        overlay_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(overlay_label)
        
        self.overlay_view = ImageLabel("Overlay will appear here")
        self.overlay_view.setMinimumHeight(300)
        right_layout.addWidget(self.overlay_view)
        
        # Statistics
        self.stats = StatsWidget()
        right_layout.addWidget(self.stats)
        
        layout.addWidget(right_panel, 1)
    
    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "",
                                               "PyTorch Model (*.pth *.pt)")
        if path:
            try:
                self.main_window.model = UNet(n_channels=3, n_classes=4)
                state_dict = torch.load(path, map_location=self.main_window.device)
                
                if 'model_state_dict' in state_dict:
                    self.main_window.model.load_state_dict(state_dict['model_state_dict'])
                else:
                    self.main_window.model.load_state_dict(state_dict)
                
                self.main_window.model = self.main_window.model.to(self.main_window.device)
                self.main_window.model.eval()
                
                self.model_status.setText(f"Loaded: {os.path.basename(path)}")
                self.model_status.setStyleSheet(f"color: {COLORS['accent_green']}; font-weight: bold;")
                self.update_analyze_button()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")
    
    def create_demo_model(self):
        try:
            self.main_window.model = UNet(n_channels=3, n_classes=4)
            self.main_window.model = self.main_window.model.to(self.main_window.device)
            self.main_window.model.eval()
            
            self.model_status.setText("Demo (untrained)")
            self.model_status.setStyleSheet(f"color: {COLORS['accent_orange']}; font-weight: bold;")
            self.update_analyze_button()
            
            QMessageBox.information(self, "Demo Model",
                "Created untrained demo model.\n\n"
                "Results will be RANDOM!\n\n"
                "Train the model first for real predictions.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create model:\n{str(e)}")
    
    def upload_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Satellite Image", "",
                                               "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        if path:
            try:
                image = Image.open(path).convert('RGB')
                self.current_image = np.array(image)
                self.current_image_path = path
                
                self.display_image(self.current_image, self.original_view)
                
                h, w = self.current_image.shape[:2]
                self.image_info.setText(f"{os.path.basename(path)}\n{w} x {h} px")
                
                # Reset results
                self.current_mask = None
                self.mask_view.clear_image()
                self.overlay_view.clear_image()
                self.save_btn.setEnabled(False)
                
                self.update_analyze_button()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")
    
    def update_analyze_button(self):
        self.analyze_btn.setEnabled(
            self.main_window.model is not None and self.current_image is not None
        )
    
    def run_analysis(self):
        if self.main_window.model is None or self.current_image is None:
            return
        
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.pred_thread = PredictionThread(
            self.main_window.model,
            self.current_image,
            self.main_window.device
        )
        self.pred_thread.finished.connect(self.on_analysis_finished)
        self.pred_thread.error.connect(self.on_analysis_error)
        self.pred_thread.progress.connect(self.progress_bar.setValue)
        self.pred_thread.start()
    
    def on_analysis_finished(self, mask, probs):
        self.current_mask = mask
        
        mask_colored = self.colorize_mask(mask)
        self.display_image(mask_colored, self.mask_view)
        
        self.update_overlay()
        self.stats.update_stats(mask)
        
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.save_btn.setEnabled(True)
    
    def on_analysis_error(self, error_msg):
        QMessageBox.critical(self, "Analysis Error", error_msg)
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
    
    def colorize_mask(self, mask):
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        colors = {
            0: (128, 128, 128),
            1: (255, 71, 87),
            2: (55, 66, 250),
            3: (46, 213, 115)
        }
        
        for class_id, color in colors.items():
            colored[mask == class_id] = color
        
        return colored
    
    def update_overlay(self):
        if self.current_image is None or self.current_mask is None:
            return
        
        alpha = self.transparency_slider.value() / 100.0
        self.transparency_label.setText(f"{int(alpha * 100)}%")
        
        mask_colored = self.colorize_mask(self.current_mask)
        overlay = (
            self.current_image.astype(float) * (1 - alpha) +
            mask_colored.astype(float) * alpha
        ).astype(np.uint8)
        
        self.current_overlay = overlay
        self.display_image(overlay, self.overlay_view)
    
    def display_image(self, image_array, label):
        h, w, c = image_array.shape
        bytes_per_line = c * w
        
        q_image = QImage(
            image_array.tobytes(),
            w, h, bytes_per_line,
            QImage.Format_RGB888
        )
        
        label.setPixmap(QPixmap.fromImage(q_image))
    
    def save_to_results(self):
        if self.current_mask is None:
            return
        
        # Save to my results tab
        self.main_window.my_results_tab.add_result(
            self.current_image,
            self.current_mask,
            self.current_overlay,
            getattr(self, 'current_image_path', 'Unknown')
        )
        
        QMessageBox.information(self, "Saved", "Result saved to 'My Results' tab!")


class MyResultsTab(QWidget):
    """Tab 3: My Results (User's analysis history)"""
    
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.results = []
        self.current_index = 0
        
        self.load_results()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Header
        header = QHBoxLayout()
        
        title = QLabel("My Results")
        title.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {COLORS['text_primary']};")
        header.addWidget(title)
        
        header.addStretch()
        
        self.count_label = QLabel("0 results")
        self.count_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 16px;")
        header.addWidget(self.count_label)
        
        layout.addLayout(header)
        
        # Navigation
        nav_layout = QHBoxLayout()
        
        self.prev_btn = ModernButton("Previous", COLORS['text_secondary'])
        self.prev_btn.clicked.connect(self.show_previous)
        nav_layout.addWidget(self.prev_btn)
        
        nav_layout.addStretch()
        
        self.page_label = QLabel("0 / 0")
        self.page_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 18px; font-weight: bold;")
        nav_layout.addWidget(self.page_label)
        
        nav_layout.addStretch()
        
        self.next_btn = ModernButton("Next", COLORS['text_secondary'])
        self.next_btn.clicked.connect(self.show_next)
        nav_layout.addWidget(self.next_btn)
        
        layout.addLayout(nav_layout)
        
        # Content
        content = QFrame()
        content.setStyleSheet(f"background: {COLORS['bg_card']}; border-radius: 20px;")
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(20)
        content_layout.setContentsMargins(30, 30, 30, 30)
        
        # Info
        self.result_info = QLabel("No results yet")
        self.result_info.setStyleSheet(f"font-size: 16px; color: {COLORS['accent_blue']};")
        content_layout.addWidget(self.result_info)
        
        # Images
        images_layout = QHBoxLayout()
        
        # Original
        orig_container = QVBoxLayout()
        orig_label = QLabel("Original")
        orig_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        orig_label.setAlignment(Qt.AlignCenter)
        orig_container.addWidget(orig_label)
        
        self.original_view = ImageLabel("Original image")
        self.original_view.setMinimumSize(300, 250)
        orig_container.addWidget(self.original_view)
        images_layout.addLayout(orig_container)
        
        # Mask
        mask_container = QVBoxLayout()
        mask_label = QLabel("Segmentation")
        mask_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        mask_label.setAlignment(Qt.AlignCenter)
        mask_container.addWidget(mask_label)
        
        self.mask_view = ImageLabel("Segmentation mask")
        self.mask_view.setMinimumSize(300, 250)
        mask_container.addWidget(self.mask_view)
        images_layout.addLayout(mask_container)
        
        # Overlay
        overlay_container = QVBoxLayout()
        overlay_label = QLabel("Overlay")
        overlay_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        overlay_label.setAlignment(Qt.AlignCenter)
        overlay_container.addWidget(overlay_label)
        
        self.overlay_view = ImageLabel("Overlay view")
        self.overlay_view.setMinimumSize(300, 250)
        overlay_container.addWidget(self.overlay_view)
        images_layout.addLayout(overlay_container)
        
        content_layout.addLayout(images_layout)
        
        # Stats
        self.stats = StatsWidget()
        content_layout.addWidget(self.stats)
        
        # Delete button
        delete_layout = QHBoxLayout()
        delete_layout.addStretch()
        
        export_btn = ModernButton("Export Images", COLORS['accent_blue'])
        export_btn.clicked.connect(self.export_current)
        delete_layout.addWidget(export_btn)
        
        self.delete_btn = ModernButton("Delete", "#ff4757")
        self.delete_btn.clicked.connect(self.delete_current)
        self.delete_btn.setVisible(False)
        delete_layout.addWidget(self.delete_btn)
        content_layout.addLayout(delete_layout)
        
        layout.addWidget(content)
        
        self.update_display()
    
    def load_results(self):
        results_file = os.path.join(self.data_dir, 'my_results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
    
    def save_results(self):
        os.makedirs(self.data_dir, exist_ok=True)
        results_file = os.path.join(self.data_dir, 'my_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
    
    def add_result(self, original, mask, overlay, source_path):
        images_dir = os.path.join(self.data_dir, 'result_images')
        os.makedirs(images_dir, exist_ok=True)
        
        result_id = str(uuid.uuid4())[:8]
        
        # Save images
        orig_path = os.path.join(images_dir, f"{result_id}_original.png")
        mask_path = os.path.join(images_dir, f"{result_id}_mask.png")
        overlay_path = os.path.join(images_dir, f"{result_id}_overlay.png")
        
        Image.fromarray(original).save(orig_path)
        
        # Colorize mask for saving
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        colors = {0: (128, 128, 128), 1: (255, 71, 87), 2: (55, 66, 250), 3: (46, 213, 115)}
        for class_id, color in colors.items():
            colored[mask == class_id] = color
        Image.fromarray(colored).save(mask_path)
        
        if overlay is not None:
            Image.fromarray(overlay).save(overlay_path)
        
        # Calculate stats
        total = mask.size
        stats = {}
        for c in range(4):
            stats[str(c)] = float(np.sum(mask == c) / total * 100)
        
        result = {
            'id': result_id,
            'source': os.path.basename(source_path),
            'original': orig_path,
            'mask': mask_path,
            'overlay': overlay_path,
            'stats': stats,
            'created': datetime.now().isoformat()
        }
        
        self.results.append(result)
        self.save_results()
        self.current_index = len(self.results) - 1
        self.update_display()
    
    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
    
    def show_next(self):
        if self.current_index < len(self.results) - 1:
            self.current_index += 1
            self.update_display()
    
    def delete_current(self):
        if not self.results:
            return
        
        reply = QMessageBox.question(self, "Delete Result",
                                     "Are you sure you want to delete this result?",
                                     QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            result = self.results[self.current_index]
            
            for key in ['original', 'mask', 'overlay']:
                path = result.get(key, '')
                if os.path.exists(path):
                    os.remove(path)
            
            del self.results[self.current_index]
            self.save_results()
            
            if self.current_index >= len(self.results):
                self.current_index = max(0, len(self.results) - 1)
            
            self.update_display()
    
    def export_current(self):
        if not self.results:
            return
        
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if folder:
            result = self.results[self.current_index]
            
            import shutil
            for key in ['original', 'mask', 'overlay']:
                src = result.get(key, '')
                if os.path.exists(src):
                    dst = os.path.join(folder, os.path.basename(src))
                    shutil.copy(src, dst)
            
            QMessageBox.information(self, "Exported", f"Images exported to:\n{folder}")
    
    def update_display(self):
        n = len(self.results)
        self.count_label.setText(f"{n} result{'s' if n != 1 else ''}")
        
        self.prev_btn.setEnabled(self.current_index > 0)
        self.next_btn.setEnabled(self.current_index < n - 1)
        self.delete_btn.setVisible(n > 0)
        
        if n == 0:
            self.page_label.setText("0 / 0")
            self.result_info.setText("No results yet. Run an analysis to see results here.")
            self.original_view.clear_image()
            self.mask_view.clear_image()
            self.overlay_view.clear_image()
            return
        
        self.page_label.setText(f"{self.current_index + 1} / {n}")
        
        result = self.results[self.current_index]
        
        created = datetime.fromisoformat(result['created']).strftime("%Y-%m-%d %H:%M")
        self.result_info.setText(f"{result.get('source', 'Unknown')}  •  {created}")
        
        if os.path.exists(result.get('original', '')):
            self.original_view.setPixmap(QPixmap(result['original']))
        else:
            self.original_view.clear_image()
        
        if os.path.exists(result.get('mask', '')):
            self.mask_view.setPixmap(QPixmap(result['mask']))
        else:
            self.mask_view.clear_image()
        
        if os.path.exists(result.get('overlay', '')):
            self.overlay_view.setPixmap(QPixmap(result['overlay']))
        else:
            self.overlay_view.clear_image()
        
        # Update stats
        stats = result.get('stats', {})
        # Create fake mask for stats widget
        if stats:
            self.stats.labels[0].setText(f"{stats.get('0', 0):.1f}%")
            self.stats.labels[1].setText(f"{stats.get('1', 0):.1f}%")
            self.stats.labels[2].setText(f"{stats.get('2', 0):.1f}%")
            self.stats.labels[3].setText(f"{stats.get('3', 0):.1f}%")
            
            max_width = 200
            for c in range(4):
                pct = stats.get(str(c), 0)
                self.stats.bars[c].setFixedWidth(int(max_width * pct / 100))


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data directory
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app_data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.init_ui()
        self.apply_style()
    
    def init_ui(self):
        self.setWindowTitle("Satellite Image Segmentation")
        self.setMinimumSize(1400, 900)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none;
                background: {COLORS['bg_dark']};
            }}
            QTabBar::tab {{
                background: {COLORS['bg_card']};
                color: {COLORS['text_secondary']};
                padding: 16px 32px;
                margin-right: 4px;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                font-weight: bold;
                font-size: 14px;
            }}
            QTabBar::tab:selected {{
                background: {COLORS['bg_dark']};
                color: {COLORS['accent_blue']};
            }}
            QTabBar::tab:hover {{
                background: {COLORS['bg_hover']};
            }}
        """)
        
        # Create tabs
        self.comparisons_tab = ComparisonsTab(self.data_dir)
        self.analysis_tab = AnalysisTab(self)
        self.my_results_tab = MyResultsTab(self.data_dir)
        
        self.tabs.addTab(self.comparisons_tab, "Comparisons")
        self.tabs.addTab(self.analysis_tab, "New Analysis")
        self.tabs.addTab(self.my_results_tab, "My Results")
        
        layout.addWidget(self.tabs)
        
        # Status bar
        self.statusBar = QStatusBar()
        self.statusBar.setStyleSheet(f"""
            QStatusBar {{
                background: {COLORS['bg_card']};
                color: {COLORS['text_secondary']};
                padding: 8px;
            }}
        """)
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage(f"Ready • Device: {self.device}")
    
    def apply_style(self):
        self.setStyleSheet(f"""
            QMainWindow {{
                background: {COLORS['bg_dark']};
            }}
            QLabel {{
                color: {COLORS['text_primary']};
            }}
            QScrollBar:vertical {{
                background: {COLORS['bg_card']};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: {COLORS['border']};
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)
