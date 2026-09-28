"""Точка входа: `python -m facade_digitizer.ui` либо `facade-digitize-ui`."""
import sys


def main(argv=None) -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as error:
        print("Интерфейс оператора требует PySide6: pip install -e \".[ui]\". "
              f"Причина: {error}", file=sys.stderr)
        return 2

    from facade_digitizer.ui.window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv if argv is None else argv)
    window = MainWindow()
    window.resize(1400, 900)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
