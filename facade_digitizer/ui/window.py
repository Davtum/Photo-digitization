"""Главное окно приложения оператора. План 3, задача 1 — каркас.

Сейчас окно пусто: холст, панели и инструменты добавляются задачами 9–18. Каркас
нужен раньше них, чтобы каждая следующая задача проверялась в настоящем окне, а
не в виджете, собранном отдельно от него.
"""
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QLabel, QMainWindow

TITLE = "Оцифровка фасада"


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(TITLE)
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.setCentralWidget(self.view)
        self.status_label = QLabel("Снимок не открыт")
        self.statusBar().addWidget(self.status_label)
