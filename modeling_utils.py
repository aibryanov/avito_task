import numpy as np
import torch as torch
from typing import List
from torch.nn.utils.rnn import pad_sequence


def prepare_tensors(labels, target, pad_value=0, ignore_value=-1):
    """
    Преобразует X, y в padded тензоры для обучения модели.

    Args:
        X (list of list[int]): список последовательностей индексов символов
        y (list of list[int]): список последовательностей меток (0/1 для пробела)
        pad_idx (int): индекс для паддинга входов
        ignore_index (int): индекс для игнорируемых токенов в loss

    Returns:
        X_padded (torch.LongTensor): (batch_size, max_len)
        y_padded (torch.LongTensor): (batch_size, max_len)
    """
    X_tensors = [torch.tensor(seq, dtype=torch.long) for seq in labels]
    y_tensors = [torch.tensor(seq, dtype=torch.long) for seq in target]

    X_padded = pad_sequence(X_tensors, batch_first=True, padding_value=pad_value)
    y_padded = pad_sequence(y_tensors, batch_first=True, padding_value=ignore_value)

    return X_padded, y_padded

def compute_f1(y_true, y_pred, ignore_index=-1):
    """
    Считает средний F1 по батчу текстов.

    Args:
        y_true (torch.Tensor): (batch, seq_len) целевые метки (0/1), паддинг = ignore_index
        y_pred (torch.Tensor): (batch, seq_len) предсказанные метки (0/1)
        ignore_index (int): индекс для паддинга, который исключаем из метрик

    Returns:
        f1 (float): средний F1 по батчу
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    f1_scores = []

    for true_seq, pred_seq in zip(y_true, y_pred):
        # Убираем паддинг
        mask = true_seq != ignore_index
        true_seq = true_seq[mask]
        pred_seq = pred_seq[mask]

        # позиции пробелов (индексы, где == 1)
        true_spaces = set(np.where(true_seq == 1)[0])
        pred_spaces = set(np.where(pred_seq == 1)[0])

        if len(true_spaces) == 0 and len(pred_spaces) == 0:
            f1_scores.append(1.0)  # оба пустые - полное совпадение
            continue
        if len(pred_spaces) == 0:
            f1_scores.append(0.0)
            continue

        precision = len(true_spaces & pred_spaces) / len(pred_spaces)
        recall = len(true_spaces & pred_spaces) / len(true_spaces) if len(true_spaces) > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

            f1_scores.append(f1)

    return float(np.mean(f1_scores))

def tokenize_text(text: str, char2id: dict) -> List[int]:
    """
    Токенизирует текст посимвольно по словарю char2id.

    Args:
        text (str): Входной текст.
        char2id (dict): Словарь символ -> id (содержит <UNK>).

    Returns:
        list[int]: Список индексов.
    """
    unk_id = char2id.get("<UNK>")
    return [char2id.get(ch, unk_id) for ch in text]

def predict_with_spaces(df, model, char2idx, device="cpu", text_column="text_no_spaces"):
    """
    Предсказывает пробелы для текстов в датафрейме.

    Args:
        df (pd.DataFrame): датафрейм с колонкой text (строка без пробелов)
        model (nn.Module): обученная модель BiLSTM
        char2idx (dict): словарь символ -> индекс
        device (str): 'cpu' или 'cuda'
        text_column (str): название колонки с текстами без пробелов

    Returns:
        pd.DataFrame: датафрейм с колонкой predicted_text
    """
    model.eval()
    predicted_texts = []

    for raw_text in df[text_column].tolist():
        # 1. Переводим строку в индексы
        x = [char2idx.get(c, 0) for c in raw_text]
        x_tensor = torch.tensor([x], dtype=torch.long).to(device)

        # 2. Прогоняем через модель
        with torch.no_grad():
            logits = model(x_tensor)  # (1, seq_len, 2)
            preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # 3. Восстанавливаем текст с пробелами
        spaced_text = ""
        for c, p in zip(raw_text, preds):
            spaced_text += c
            if p == 1:
                spaced_text += " "
        predicted_texts.append(spaced_text.strip())

    df = df.copy()
    df["predicted_text"] = predicted_texts
    return df
