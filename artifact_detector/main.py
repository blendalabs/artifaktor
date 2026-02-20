"""Artifaktor — Visual artifact detector for AI-generated animation frames."""

import sys


def main():
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print(
            "PySide6 is required. Install dependencies with:\n"
            "  pip install -r requirements.txt\n"
            "or:\n"
            "  uv pip install PySide6 opencv-python-headless numpy"
        )
        sys.exit(1)

    from .ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Artifaktor")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
