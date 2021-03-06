import tkinter
from tkinter import *


def send():
   msg = EntryBox.get("1.0",'end-1c').strip()
   EntryBox.delete("0.0",END)

   if msg != '':
       ChatLog.config(state=NORMAL)
       ChatLog.insert(END, "You: " + msg + '\n\n')
       ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

       res = chatbot_response(msg)
       ChatLog.insert(END, "Bot: " + res + '\n\n')

       ChatLog.config(state=DISABLED)
       ChatLog.yview(END)

base = Tk()
base.title("Hola")
base.geometry("400*500")
base.resizable(0*0)

Chatlog = Text(base, bd=0, bg = "white", height = "8", width = "50", font = "Arial")
Chatlog.config(state = DISABLED)


scrollbar = Scrollbar(base, command = ChatLog.yview, cursor = "heart")
Chatlog['yscrollcommand'] = scrollbar.set

button = Button(base, font = ("Verdana", 12, 'bold'), text = "Send", width = "12", height = "6" , bg = "#32de97", command = send)


Entrybox = Text(base, bg = "white", width = "29", height = "6", font = "Verdana")
Entrybox.bind = ("<Return>", send)

scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()



