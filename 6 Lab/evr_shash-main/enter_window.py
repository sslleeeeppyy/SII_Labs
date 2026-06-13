import tkinter as tk
from pass_manager import Manager
from register_window import RegisterWindow
from tkinter import messagebox


class EnterWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.manager = Manager()
        self.response = False

        # Создание и настройка окна авторизации
        self.title('Авторизация')
        self.geometry('450x330')
        self.resizable(False, False)
        font_header = ('Arial', 15)
        font_entry = ('Arial', 12)
        label_font = ('Arial', 11)
        base_padding = {'padx': 10, 'pady': 8}
        header_padding = {'padx': 10, 'pady': 12}

        # Обработка нажатия кнопки "Войти"
        def clicked_auth():
            username = username_entry.get()
            password = password_entry.get()
            if self.manager.get_user(username, password):
                self.response = True
                self.destroy()
            else:
                messagebox.showerror('Ошибка', 'Неверно введен пароль или логин')

        # Обработка нажатия кнопки "Регистрация"
        def clicked_register():
            register_window = RegisterWindow()

        main_label = tk.Label(self, text='Авторизация', font=font_header, justify=tk.CENTER, **header_padding)
        main_label.pack()
        # Метка для поля ввода имени
        username_label = tk.Label(self, text='Имя пользователя', font=label_font, **base_padding)
        username_label.pack()
        # Поле ввода имени
        username_entry = tk.Entry(self, bg='#fff', fg='#444', font=font_entry)
        username_entry.pack()
        # Метка для поля ввода пароля
        password_label = tk.Label(self, text='Пароль', font=label_font, **base_padding)
        password_label.pack()
        # Поле ввода пароля
        password_entry = tk.Entry(self, bg='#fff', fg='#444', font=font_entry)
        password_entry.pack()
        # Кнопка отправки формы
        send_btn1 = tk.Button(self, text='Войти', command=clicked_auth)
        send_btn1.pack(**base_padding)
        # Кнопка отправки регистрации
        send_btn2 = tk.Button(self, text='Регистрация', command=clicked_register)
        send_btn2.pack(**base_padding)
