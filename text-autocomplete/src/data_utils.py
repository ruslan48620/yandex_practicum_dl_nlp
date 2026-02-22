import re
from pathlib import Path

import pandas as pd

URL_RE = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE = re.compile(r'@[A-Za-z0-9_]+')
NON_TEXT_RE = re.compile(r'[^a-z0-9\s\'.,!?-]')
SPACES_RE = re.compile(r'\s+')


def read_raw_dataset(path):
    '''
    Чтение сырого датасета в формат DataFrame
    '''
    path = Path(path)
    lines = [line.strip() for line in path.read_text(encoding='utf-8').splitlines()]
    df = pd.DataFrame({'raw_text': lines})
    return df


def clean_text(text):
    '''
    Очистка текста
    '''
    text = text.lower() # приведение к нижнему регистру
    text = URL_RE.sub(' ', text) # удаление ссылок
    text = MENTION_RE.sub(' ', text) # удаление упоминаний
    text = NON_TEXT_RE.sub(' ', text) # удаление нестандартных символов
    text = SPACES_RE.sub(' ', text).strip() # удаление множества подряд идущих пробелов
    return text


def process_dataset(raw_df):
    '''
    Обработка сырых текстов
    '''
    df = raw_df.copy()
    # чистим данные
    df['clean_text'] = df['raw_text'].map(clean_text)
    # оставляем только строки, где количество слов больше 2-х
    # нужно для формирования датасета
    df = df[df['clean_text'].str.split().str.len() > 2].reset_index(drop=True)
    return df
