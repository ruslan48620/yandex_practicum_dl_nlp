import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from src.eval_lstm import split_prefix_target, compute_rouge_scores


def split_prefix_target_words(text, prefix_ratio=0.75):
    words = text.split()
    prefix_words, target_words = split_prefix_target(words, prefix_ratio=prefix_ratio)

    prefix = ' '.join(prefix_words)
    target = ' '.join(target_words)
    return prefix, target


def create_distilgpt2_generator(model_name='distilgpt2', device='cpu'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    model.generation_config.do_sample = True
    model.generation_config.top_k = 50
    model.generation_config.top_p = 0.95
    model.generation_config.temperature = 0.9
    model.generation_config.repetition_penalty = 1.1
    model.generation_config.num_return_sequences = 1
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.max_length = None

    generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    return generator, tokenizer


def build_generation_params(prefix, target, tokenizer, **generation_kwargs):
    target_len_tokens = len(tokenizer.encode(target, add_special_tokens=False))
    default_max_new_tokens = max(3, min(64, target_len_tokens))

    params = {
        'max_new_tokens': default_max_new_tokens,
    }
    params.update(generation_kwargs)

    return prefix, params


@torch.no_grad()
def evaluate_transformer_rouge(generator,
                               tokenizer,
                               texts,
                               print_examples=0,
                               **generation_kwargs):
    '''
    Оценка качества генерации трансформера.
    По первым 3/4 исходного текста модель дополняет оставшиеся 1/4.
    '''
    predictions = []
    references = []
    examples_printed = 0

    model = generator.model
    model.eval()

    for text in texts:
        prefix, target = split_prefix_target_words(text)

        prompt, gen_params = build_generation_params(
            prefix=prefix,
            target=target,
            tokenizer=tokenizer,
            **generation_kwargs,
        )

        enc = tokenizer(prompt, return_tensors='pt')
        enc = {k: v.to(model.device) for k, v in enc.items()}
        generated_ids = model.generate(**enc, **gen_params)
        full_generated = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        if full_generated.startswith(prefix):
            pred_text = full_generated[len(prefix):].strip()
        else:
            pred_text = full_generated.strip()

        ref_text = target.strip()

        predictions.append(pred_text)
        references.append(ref_text)

        if print_examples > 0 and examples_printed < print_examples:
            print(f'prefix: {prefix}')
            print(f'target: {ref_text}')
            print(f'pred  : {pred_text}')
            print('-' * 60)
            examples_printed += 1

    scores = compute_rouge_scores(predictions, references)

    print(
        f'rouge1={scores["rouge1"]:.4f} | '
        f'rouge2={scores["rouge2"]:.4f}'
    )

    return {
        'rouge1': scores['rouge1'],
        'rouge2': scores['rouge2'],
        'n_samples': len(predictions),
    }
