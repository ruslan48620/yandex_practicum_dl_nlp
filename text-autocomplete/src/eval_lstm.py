import torch
import evaluate

ROUGE = evaluate.load('rouge')


def reconstruct_full_ids(input_ids, labels, length):
    '''
    Восстановление исходной последовательности токенов без сдвигов
    '''
    valid_input = input_ids[:length].tolist()

    # последний токен из последовательности хранится в labels[length - 1].
    last_token = int(labels[length - 1].item())
    if last_token == -100:
        return valid_input
    return valid_input + [last_token]


def split_prefix_target(token_ids, prefix_ratio=0.75):
    prefix_len = max(1, int(len(token_ids) * prefix_ratio))
    prefix_len = min(prefix_len, len(token_ids) - 1)

    prefix = token_ids[:prefix_len]
    target = token_ids[prefix_len:]
    return prefix, target


def compute_rouge_scores(predictions, references):
    results = ROUGE.compute(predictions=predictions, references=references)

    return {
        'rouge1': results['rouge1'],
        'rouge2': results['rouge2'],
    }


@torch.no_grad()
def evaluate_lstm_rouge(model,
                        loader,
                        tokenizer,
                        device='cpu',
                        print_examples=0):
    '''
    Оценка качества генерации на валидации
    По первым 3/4 исходного текста модель дополняет оставшиеся 1/4
    '''
    model.eval()

    predictions = []
    references = []
    examples_printed = 0

    for  batch in loader:
        ids, labels, lengths = batch['input_ids'], batch['labels'], batch['lengths']

        batch_size = ids.size(0)
        for i in range(batch_size):
            length = int(lengths[i].item())
            full_ids = reconstruct_full_ids(ids[i], labels[i], length)

            prefix_ids, target_ids = split_prefix_target(full_ids)
            if not target_ids:
                continue

            generated = model.generate(
                prefix_ids=torch.tensor(prefix_ids, dtype=torch.long, device=device),
                max_new_tokens=len(target_ids),
            )

            generated_ids = generated.detach().cpu().tolist()
            pred_completion = generated_ids[len(prefix_ids):]

            pred_text = tokenizer.decode(pred_completion, skip_special_tokens=True).strip()
            ref_text = tokenizer.decode(target_ids, skip_special_tokens=True).strip()

            predictions.append(pred_text)
            references.append(ref_text)

            if print_examples > 0 and examples_printed < print_examples:
                prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True).strip()
                print(f'prefix: {prefix_text}')
                print(f'target: {ref_text}')
                print(f'pred  : {pred_text}')
                print('-' * 60)
                examples_printed += 1

    return compute_rouge_scores(predictions, references)
