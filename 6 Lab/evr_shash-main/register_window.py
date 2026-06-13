import tkinter as tk
from pass_manager import Manager
from tkinter import messagebox


class RegisterWindow(tk.Tk):

    def __init__(self):
        super().__init__()
        self.manager = Manager()

        # Создание и настройка окна
        self.title('Регистрация')
        self.geometry('450x330')
        self.resizable(False, False)
        font_header = ('Arial', 15)
        font_entry = ('Arial', 12)
        label_font = ('Arial', 11)
        base_padding = {'padx': 10, 'pady': 8}
        header_padding = {'padx': 10, 'pady': 12}

        # обработчик нажатия на клавишу 'Регистрация'
        def clicked_register():
            password1 = password_entry_1.get()
            password2 = password_entry_2.get()
            if len(password1) == 0:
                return
            if password2 != password1:
                messagebox.showerror('Ошибка', 'Пароли не совпадают')
                return
            username = username_entry.get()
            if self.manager.add_user(username, password1):
                messagebox.showinfo('Успешно', 'Пользователь добавлен')
                self.destroy()
            else:
                messagebox.showerror('Ошибка', 'Измените имя пользователя или пароль')

        main_label = tk.Label(self, text='Регистрация', font=font_header, justify=tk.CENTER, **header_padding)
        main_label.pack()
        # метка для поля ввода имени
        username_label = tk.Label(self, text='Имя пользователя', font=label_font, **base_padding)
        username_label.pack()
        # поле ввода имени
        username_entry = tk.Entry(self, bg='#fff', fg='#444', font=font_entry)
        username_entry.pack()
        # метка для поля ввода пароля
        password_label_1 = tk.Label(self, text='Пароль', font=label_font, **base_padding)
        password_label_1.pack()
        # поле ввода пароля
        password_entry_1 = tk.Entry(self, bg='#fff', fg='#444', font=font_entry)
        password_entry_1.pack()
        # метка для поля ввода пароля
        password_label_2 = tk.Label(self, text='Повторите пароль', font=label_font, **base_padding)
        password_label_2.pack()
        # поле ввода пароля
        password_entry_2 = tk.Entry(self, bg='#fff', fg='#444', font=font_entry)
        password_entry_2.pack()
        # кнопка отправки регистрации
        send_btn = tk.Button(self, text='Зарегистрироваться', command=clicked_register)
        send_btn.pack(**base_padding)
        # запускаем главный цикл окна
        #self.window.mainloop()
