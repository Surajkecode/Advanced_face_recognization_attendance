from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, Toplevel, Text, Entry
import subprocess

def login():
    username = "facesnap"
    password = "facesnap"

    if not username_entry.get() or not password_entry.get():
        messagebox.showwarning(title="Warning", message="Please fill in both fields.")
        return

    if username_entry.get() == username and password_entry.get() == password:
        messagebox.showinfo(title="Login Success", message="You successfully logged in.")
        main_py_path = r"E:\face Advanced\FaceAdvanced\Attendanceface (1)\Attendanceface\main.py"
        subprocess.Popen(['python', main_py_path])
    else:
        messagebox.showerror(title="Error", message="Invalid login.")

def chatbot():
    chat_window = Toplevel(window)
    chat_window.title("Chatbot")
    chat_window.geometry("300x300")  # Set size of chatbot window

    chat_window.grid_rowconfigure(0, weight=1)  # Allow chat_text to expand
    chat_window.grid_rowconfigure(1, weight=0)  # Fixed row for chat_entry
    chat_window.grid_columnconfigure(0, weight=1)  # Allow chat_entry to expand horizontally

    chat_text = Text(chat_window, wrap='word', bg='#F5F5F5', font=("Helvetica", 12))
    chat_text.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)  # Sticky to expand

    chat_entry = Entry(chat_window, font=("Helvetica", 12), bd=2, relief='flat')
    chat_entry.grid(row=1, column=0, sticky='ew', padx=10, pady=10)  # Sticky to expand horizontally

    def handle_chat(event=None):
        user_input = chat_entry.get().strip().lower()
        chat_entry.delete(0, 'end')

        chat_text.insert('end', f'You: {user_input}\n')

        # Response handling
        if "hi" in user_input or "hello" in user_input:
            response = "Welcome Tgpcet, how can I help you?"
        elif "forgot password" in user_input or "forget password" in user_input:
            response = " Enter your college ID, and I'll assist you in resetting your password."
        elif "forgot username" in user_input or "forget username" in user_input:
            response = "Enter your college ID, and I'll assist you in retrieving your username."
        else:
            response = "Could you please provide your college ID? The administrator will be in touch shortly.”."

        chat_text.insert('end', f'Bot: {response}\n')
        chat_text.see('end')

    chat_entry.bind('<Return>', handle_chat)
    chat_entry.focus()

window = tk.Tk()
window.title("Login Form")
window.geometry('1920x1080')
window.configure(bg='#F5F5F5')

background_image = Image.open(r"E:\face Advanced\FaceAdvanced\Attendanceface (1)\Attendanceface\login.jpg")
background_photo = ImageTk.PhotoImage(background_image)

background_label = tk.Label(window, image=background_photo)
background_label.place(relwidth=1, relheight=1)

heading_label = tk.Label(window, text="TGPCET FACESNAP", bg='#F8F8F8', fg="#FF3399", font=("Helvetica", 30, 'bold italic'))
heading_label.pack(pady=20)

frame_width = 400
frame_height = 400
frame = tk.Frame(window, bg='#FFFFFF', padx=20, pady=20, bd=2, relief='flat', width=frame_width, height=frame_height)
frame.place(relx=0.5, rely=0.5, anchor='center')

login_label = tk.Label(frame, text="Login", bg='#FFFFFF', fg="#FF3399", font=("Helvetica", 30, 'bold'))
username_label = tk.Label(frame, text="Username", bg='#FFFFFF', fg="#333333", font=("Helvetica", 16))
username_entry = tk.Entry(frame, font=("Helvetica", 16), bd=2, relief='flat', bg='#F0F0F0', highlightbackground='#DDDDDD', highlightcolor='#FF3399')

password_label = tk.Label(frame, text="Password", bg='#FFFFFF', fg="#333333", font=("Helvetica", 16))
password_entry = tk.Entry(frame, show="*", font=("Helvetica", 16), bd=2, relief='flat', bg='#F0F0F0', highlightbackground='#DDDDDD', highlightcolor='#FF3399')

login_button = tk.Button(frame, text="Login", bg="#FF3399", fg="#FFFFFF", font=("Helvetica", 16, 'bold'), command=login, relief='flat', bd=0, highlightthickness=0, padx=20, pady=10)

forgot_password_button = tk.Button(frame, text="Forgot Password?", bg="#FFFFFF", fg="#FF3399", font=("Helvetica", 12), command=lambda: messagebox.showinfo(title="Forgot Password", message="Please contact the administrator to reset your password."), relief='flat', bd=0)

login_label.grid(row=0, column=0, columnspan=2, pady=20)
username_label.grid(row=1, column=0, sticky='e', padx=10)
username_entry.grid(row=1, column=1, pady=10, padx=10, ipadx=5)
password_label.grid(row=2, column=0, sticky='e', padx=10)
password_entry.grid(row=2, column=1, pady=10, padx=10, ipadx=5)
login_button.grid(row=3, column=0, columnspan=2, pady=20)
forgot_password_button.grid(row=4, column=0, columnspan=2, pady=10)

chatbot_icon = Image.open(r"E:\face Advanced\FaceAdvanced\Attendanceface (1)\Attendanceface\chatbot.png")
chatbot_icon = chatbot_icon.resize((100, 100), Image.LANCZOS)  # Resize the icon to a smaller size
chatbot_photo = ImageTk.PhotoImage(chatbot_icon)

chatbot_button = tk.Button(window, image=chatbot_photo, command=chatbot, relief='flat', bg='#F5F5F5', bd=0)
chatbot_button.place(relx=0.8, rely=0.8, anchor='sw')  # Adjusted position

window.mainloop()
