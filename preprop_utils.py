import pandas as pd
import numpy as np
from typing import List, Tuple
import re


def file_to_df(path: str, format: str) -> pd.DataFrame:
    """
    Считать файл в датафрейм

      path: file path
      format:
          txt - .txt format with 2 columns (id,text)
          csv - .csv format with coma separation
          parquet - .parquet format

      return: pd.DataFrame
    """
    if format == 'txt':
        rows = []
        with open(path, encoding="utf-8") as f:
            header = next(f).strip().split(",", 1)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                id, text = line.split(",", 1)  # делим только по первой запятой
                rows.append((int(id), text))

        return pd.DataFrame(rows, columns=header)

    elif format == 'csv':
        data = pd.read_csv(path)

    elif format == 'parquet':
        data = pd.read_parquet(path)

    else:
        print('Введите один из форматов: csv, txt, parquet!')

    return data

def preprocess_lyrics(text: str) -> str:
    """Уберу квадратные скобки и то что внутри них"""

    clean_text = re.sub(r'\[[^\]]*\]', '', text)
    return clean_text.strip()

def sep_to_segments(text: str, min_words: int = 4, max_words: int = 10) -> List[str]:
    """
    Разбивает текст на непересекающиеся сегменты с заданным количеством слов.

      text: исходный текст
      min_words: минимальное количество слов в сегменте
      max_words: максимальное количество слов в сегменте

      return: список сегментов текста
    """
    words_list = text.split()
    segments = []
    idx = 0

    while idx < len(words_list):
        seg_length = np.random.randint(min_words, max_words + 1)
        end_idx = idx + seg_length
        if end_idx > len(words_list):
            break  # если сегмент выходит за границы, останавливаем
        segment = " ".join(words_list[idx:end_idx])
        segments.append(segment)
        idx = end_idx  # смещаем индекс на длину сегмента

    return segments

def build_char_vocab(texts: List[str]) -> dict:
    """
    Строит словарь символов

        texts: список текстов

        return: dict {символ: индекс}
    """
    # все символы во всех текстах
    chars = sorted(set("".join(texts)))

    char2id = {ch: idx + 1 for idx, ch in enumerate(chars)}

    # добавляем спецтокены
    char2id["<PAD>"] = 0
    char2id["<UNK>"] = len(char2id)

    return char2id

def build_target(text: str) -> Tuple[str, List[int]]:
    """
    Строит вектор таргета и текст без пробелов

        text: текст с пробелами

        return: текст без пробелов, вектор таргета
    """
    words = text.split()
    clean_text = "".join(words)
    labels = []

    for word in words:
        if not word:
            continue
        # последний = 1
        labels.extend([0] * (len(word) - 1))
        labels.append(1)

    if labels:
        labels[-1] = 0

    return clean_text, labels

def mark_word_starts(text: str) -> Tuple[str, List[int]]:
    """
    Создать вектор таргета и текст без пробелов
    Единица = начало слова, остальные символы = 0

        text: текст с пробелами

        return: (текст без пробелов, список меток)
    """
    words = text.split()
    clean_text = "".join(words)
    labels = []

    for word in words:
        if not word:
            continue
        # первый символ слова = 1, остальные = 0
        labels.append(1)
        labels.extend([0] * (len(word) - 1))
    labels[0] = 0

    return clean_text, labels
