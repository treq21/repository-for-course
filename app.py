import re
import tkinter as tk
import tkinter.filedialog as fd
import warnings
import keras
import numpy as np
import pandas as pd
import pymorphy2
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

warnings.filterwarnings('ignore')




model_loaded = keras.models.load_model('lstm_model.h5')
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('stopwords')

stop = set(stopwords.words("russian"))
stop_en = set(stopwords.words("english"))
stop.update(['добрый','день','день'])
stop.update(stop_en)  
ma = pymorphy2.MorphAnalyzer()
def remove_stopwords(text):
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop and len(word)>2 and '@' not in word]
    return " ".join(filtered_words)
def clean_text(text):
    text = text.replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
    text = text.lower()
    text = re.sub('!_№\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text) #deleting newlines and line-breaks
    text = re.sub('№[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text) #deleting symbols  
    text = " ".join(ma.parse(word)[0].normal_form for word in text.split())
    text = ' '.join(word for word in text.split() if len(word)>3)
    text = re.sub(r'[0-9]+', '', text)
    text = text.replace("_", " ")
    text = text.replace("!", " ")
    text = text.replace(".", " ")
    text = text.replace(",", " ")
    text = text.replace("№", " ")
    text = text.replace("…", " ")
    text = text.replace(":", " ")
    text = text.replace("&lt;", " ")
    text = text.replace(" pi ", " ")
    text = re.sub(r'\([^()]*\)', '', text)
    text = text.replace("td>", " ")
    text = text.replace("<td>", " ")
    text = text.replace("<tr>", " ")
    text = text.replace("<table>", " ")
    text = text.replace("<", " ")
    text = text.replace(">", " ")
    text = text.replace("<table>", " ")
    text = text.replace("<td>", " ")
    text = text.replace("<tr>", " ")
    text = text.replace("< td>", " ")
    text = text.replace("td>", " ")
    text = text.replace("< tr", " ")
    text = text.replace("< table>", " ")
    text = text.replace("<", " ")
    text = text.replace(">", " ")
    text = text.replace(" tr ", " ")
    text = text.replace(" ru ", " ")
    text = text.replace("{pi}", " ")
    text = text.replace(" тс ", " тсс ")
    text = text.replace("ru&gt", " ")
    text = text.replace(";", " ")
    text = re.sub(r'[\(\)]', ' ', text)
    text = text.replace("?", " ")
    text = text.replace("#", " ")
    text = text.replace("»", " ")
    text = text.replace("«", " ")
    
    
    
    text = remove_stopwords(text)


    return text




vocab_size = 6000 
embedding_dim = 64
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'




class App(tk.Tk):
    def __init__(self):
        super().__init__()
        btn_file = tk.Button(self, text="Выбрать файл",
                             command=self.choose_file)
        
        btn_file.pack(padx=60, pady=10)
        

    def choose_file(self):
        filetypes = (("Excel", "*.xlsx"),
                     
                     ("Любой", "*"))
        filename = fd.askopenfilename(title="Открыть файл", initialdir="/",
                                      filetypes=filetypes)

        df = pd.read_excel(filename)
        df_analyse = df.drop(df.columns.difference(['Текст заявки']), 1)
        # Загрузка данных
        

        df_analyse['text'] = " "
        # очистка текста при помощи функции очистки
        df_analyse['text'] = df.apply(lambda x: clean_text(x['Текст заявки']), axis=1)
        df_analyse['text'] = df_analyse.apply(lambda x: (x['text'].replace("'", " ")), axis=1)
        
#         обрезаем заявки
        for i in range(len(df_analyse.text)):
            df_analyse.text[i] = " ".join(df_analyse.text[i].split()[:max_length + 1])
            
        articles = list(df_analyse.text)
        tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
        x = list(pd.read_csv('learn_tokinisator.csv')['text'])
        tokenizer.fit_on_texts(x)
        word_index = tokenizer.word_index
        
        articles = tokenizer.texts_to_sequences(articles)
        articles = pad_sequences(articles, maxlen=max_length, padding=padding_type, truncating=trunc_type, dtype='float32')


        classes = pd.read_csv('classes.csv')['0']

        category = []
        for i in model_loaded.predict(articles):
          category.append(np.argmax(i))
        category_text = []
        for i in category:
            category_text.append(classes[i])

        df['category'] = category_text
        class App(tk.Tk):
            def __init__(self):
                super().__init__()
        
                btn_dir = tk.Button(self, text="Выбрать папку",
                             command=self.choose_directory)
        
                btn_dir.pack(padx=60, pady=10)

            def choose_directory(self):
                directory = fd.askdirectory(title="Открыть папку", initialdir="/")
                if directory:
                    df.to_csv(directory + '/result.csv')
                    print(directory + '/result.csv')

        if __name__ == "__main__":
            app = App()
            app.mainloop()

        

        

    

if __name__ == "__main__":
    app = App()
    app.mainloop()