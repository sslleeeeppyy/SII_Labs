from encode import Coder
import os


class Manager:
    def __init__(self):
        self.coder = Coder()
        self.file_name = 'users.txt'
        self.init_file()

    def init_file(self):  # Инициализация файла, если этого не сделать програма вылетит м ошибкой, что файла нет
        """Создает файл пользователей"""
        if not os.path.exists(self.file_name):
            with open('users.txt', 'w'):
                pass

    def add_user(self, login: str, password: str) -> bool:
        """Добавляет пользователя в файл"""
        with open(self.file_name, 'r') as f:
            users = f.read().splitlines()  # Считываем всех пользователей из файла

        for user in users:
            args = user.split(':')
            if login == args[
                0]:  # Если логин уже есть, парль не проверяем, шанс взлома увеличится(кто-то мб узнает пароль)
                return False  # Тут можно написать что угодно, будь то HTML статус(409 - conflict), либо просто фразу ошибки

        result = self.coder.encode(password)
        with open('users.txt', 'a') as f:
            f.write(f'{login}:{result}\n')  # Добавляем нового пользователя
        return True

    def get_user(self, login: str, password: str) -> bool:
        """Проверяет логин и пароль пользователя"""
        with open('users.txt', 'r') as f:
            users = f.read().splitlines()  # Считываем всех пользователей из файла

        result = self.coder.encode(password)
        for user in users:
            args = user.split(':')
            if login == args[0] and result == args[1]:  # Если пользователь с таким логином и паролем существует
                return True
        return False
