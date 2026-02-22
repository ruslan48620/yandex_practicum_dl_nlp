import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.eval_lstm import evaluate_lstm_rouge


def move_batch_to_device(batch, device):
    '''
    Перенос batch на device
    '''
    ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    if torch.is_tensor(batch['lengths']):
        lengths = batch['lengths'].to(dtype=torch.int64, device='cpu')
    else:
        lengths = torch.tensor(batch['lengths'], dtype=torch.int64, device='cpu')
    return ids, labels, lengths


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    '''
    Обучение на одной эпохе
    '''
    model.train()
    total_loss = 0.0
    for batch in loader:
        ids, labels, lengths = move_batch_to_device(batch, device)

        optimizer.zero_grad()
        logits = model(ids, lengths)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    '''
    Замер лосса на валидации
    '''
    model.eval()
    total_loss = 0.0

    for batch in loader:
        ids, labels, lengths = move_batch_to_device(batch, device)

        logits = model(ids, lengths)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        total_loss += loss.item()
    return total_loss / len(loader)


def train_lstm_model(model,
                     train_loader,
                     val_loader,
                     tokenizer,
                     optimizer,
                     criterion,
                     num_epochs=5,
                     device='cpu',
                     print_examples=3,
                     ):
    '''
    Обучение модели
    '''
    model = model.to(device)
    history = []

    for epoch in tqdm(range(num_epochs)):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_loss = evaluate_loss(model, val_loader, criterion, device)

        rouge_scores = evaluate_lstm_rouge(
            model=model,
            loader=val_loader,
            tokenizer=tokenizer,
            device=device,
            print_examples=print_examples if epoch % 1 == 0 else 0,
        )

        row = {
            'epoch': epoch+1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
        }
        history.append(row)

        print(
            f'epoch {epoch+1}/{num_epochs} | '
            f'train_loss={train_loss:.4f} | '
            f'val_loss={val_loss:.4f} | '
            f'rouge1={rouge_scores["rouge1"]:.4f} | '
            f'rouge2={rouge_scores["rouge2"]:.4f}'
        )

    return history


def plot_training_history(df):
    '''
    Вывод графиков функции потерь и функции качества при обучении
    '''
    # --- Loss ---
    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- ROUGE ---
    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['rouge1'], label='ROUGE-1')
    plt.plot(df['epoch'], df['rouge2'], label='ROUGE-2')
    plt.xlabel('Epoch')
    plt.ylabel('ROUGE')
    plt.title('ROUGE over epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
