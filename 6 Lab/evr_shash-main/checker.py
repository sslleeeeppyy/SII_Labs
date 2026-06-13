class Checker:

    def __init__(self, color, row, col):
        # Инициализация атрибутов шашки
        self.color = color
        self.row = row
        self.col = col
        self.is_queen = False

    def move(self, new_row, new_col):
        # Перемещение шашки на новые координаты
        self.row = new_row
        self.col = new_col
