import sys
import os

# Dodaj ścieżkę projektu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from app.main_window import MainWindow


def main():
    # Włącz skalowanie DPI
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    
    # Ustaw font aplikacji
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Ustaw styl
    app.setStyle("Fusion")
    
    # Utwórz i pokaż główne okno
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

