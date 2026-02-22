import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class NextTokenDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.encodings = []
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # токенизируем каждую строку и сохраняем последовательности токенов
        for line in texts:
            token_ids = tokenizer.encode(line, add_special_tokens=False, return_tensors='pt')
            self.encodings.append(token_ids.squeeze(0))

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        tokens = self.encodings[idx]
        return {
            'input_ids': tokens[:-1],
            'labels': tokens[1:],
            'pad_token_id': self.pad_token_id,
        }


def collate_fn(batch):
    '''
    Кастомная collate_fn: паддинг до максимальной длины в батче,
    '''
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    pad_token_id = batch[0].get('pad_token_id', 0)
    lengths = torch.tensor([len(seq) for seq in input_ids])

    # паддинг входов токеном pad из токенизатора.
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    # паддинг целей значением -100, чтобы CrossEntropyLoss игнорировал эти позиции.
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        'input_ids': padded_input_ids,
        'labels': padded_labels,
        'lengths': lengths,
    }
