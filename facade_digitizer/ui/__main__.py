"""Точка входа: `python -m facade_digitizer.ui` либо `facade-digitize-ui`.

`--operator op-xxxxxxxx` — непрозрачная метка оператора, назначенная исследованием
(план 3, задача 22); без неё метка создаётся случайной на запуск. Имя сюда не
вводится: в выходном файле оно было бы персональными данными (п. 14).
"""
import argparse
import os
import sys


def _install_russian_qt(app) -> None:
    """Стандартные диалоги Qt (открыть файл, да/нет) — по-русски, как и всё окно."""
    from PySide6.QtCore import QLibraryInfo, QTranslator

    translator = QTranslator(app)
    path = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    if translator.load("qtbase_ru", path):
        app.installTranslator(translator)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="facade-digitize-ui",
                                     description="Интерфейс оператора оцифровки фасада.")
    parser.add_argument("--operator", default=os.environ.get("FACADE_OPERATOR"),
                        help="метка оператора вида op-1a2b3c4d (или FACADE_OPERATOR)")
    parser.add_argument("--maximized", action="store_true",
                        help="окно на весь экран (так запускается в контейнере)")
    args, qt_args = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as error:
        print("Интерфейс оператора требует PySide6: pip install -e \".[ui]\". "
              f"Причина: {error}", file=sys.stderr)
        return 2

    from facade_digitizer.ui.labour import checked_label
    from facade_digitizer.ui.window import MainWindow

    if args.operator is not None:
        try:
            checked_label(args.operator)
        except ValueError as error:
            print(error, file=sys.stderr)
            return 2
    app = QApplication.instance() or QApplication([sys.argv[0], *qt_args])
    _install_russian_qt(app)
    window = MainWindow()
    if args.operator is not None:
        window.session.labour.operator = args.operator
    window.resize(1400, 900)
    if args.maximized:
        window.showMaximized()
    else:
        window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
