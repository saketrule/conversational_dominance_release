
import pandas as pd
import numpy as np
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.nn.functional import pad

def perplexity_of_fixedlength_models(dialog):
    
    max_length = model.config.n_positions
    stride = 1
    
    pad_token_id = 0
    encodings = tokenizer(" ".join(dialog), return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    padding_len = max_length -1 
    padded_input_ids = pad(torch.tensor([], dtype=torch.long), (0, padding_len), value=pad_token_id).unsqueeze(dim=0)
    encodings.input_ids = torch.cat([padded_input_ids, encodings.input_ids], dim=1)
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    prev_end_loc = padding_len
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from max_length on the last loop 
        begin_loc = max(padding_len, begin_loc)
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood.item())

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return nlls

# Assuming 'matches', 'dialog', 'offset', and 'perpl' are defined earlier in your code
def perplexity_to_info(dialog, perpl, answers):
    pattern = r'\[([^\]]+)\]'
    matches = re.findall(pattern, "".join(dialog))
    unique_matches = np.unique(matches)

    encodings = tokenizer(" ".join(dialog), return_tensors="pt")
    tokens_ids_per_sentence = np.cumsum([tokenizer(p, return_tensors="pt").input_ids.size(1) for p in dialog])
    
    ppl_to_info = {}
    prev_idx_pp = 0
    
    for idx, (match,answer) in enumerate(zip(matches,answers)):
        if match not in ppl_to_info:
            ppl_to_info[match] = {}
        if answer not in ppl_to_info[match]:
            ppl_to_info[match][answer] = []
        
        idx_pp = tokens_ids_per_sentence[idx]
        patt = matches[idx]
        label = re.sub(r'\[([^\]]+)\]: ', '', dialog[idx])
        tokens = encodings.input_ids[0,prev_idx_pp:idx_pp]
        decoded = [tokenizer.decode([token], skip_special_tokens=True) for token in tokens]
        perpl_per_sent = perpl[prev_idx_pp:idx_pp]
        mean_value=np.nanmean(np.asarray(perpl_per_sent))
        prev_idx_pp=idx_pp
        ppl_to_info[match][answer].append(np.nanmean(np.asarray(perpl_per_sent)))
    
    return ppl_to_info

            
def compute_graph_perplexity(dialog, perpl, answers=None):
    pattern = r'\[([^\]]+)\]'
    matches = re.findall(pattern, "".join(dialog))
    unique_matches = np.unique(matches)
    
    rows = int(np.ceil(np.sqrt(len(dialog))))
    # Create an 8x8 grid of subplots
    fig, axes = plt.subplots(rows, rows, figsize=(30, 30))
    num_plots = len(dialog)
    # Set smaller font size
    plt.rcParams.update({'font.size': 8})

    encodings = tokenizer(" ".join(dialog), return_tensors="pt")
    tokens_ids_per_sentence = np.cumsum([tokenizer(p, return_tensors="pt").input_ids.size(1) for p in dialog])

    prev_idx_pp = 0
    for idx, ax in enumerate(axes.flatten()):
        if idx < num_plots:
            idx_pp = tokens_ids_per_sentence[idx]
            patt = matches[idx]
            label = re.sub(r'\[([^\]]+)\]: ', '', dialog[idx])
            tokens = encodings.input_ids[0,prev_idx_pp:idx_pp]
            decoded = [tokenizer.decode([token], skip_special_tokens=True) for token in tokens]
            perpl_per_sent = perpl[prev_idx_pp:idx_pp]
            ax.plot(np.asarray(perpl_per_sent), label=f'{patt}')
            mean_value=np.nanmean(np.asarray(perpl_per_sent))
            ax.axhline(mean_value, color='r', label=f':{mean_value:.2f}')  # Fixed the color argument
            ax.set_xticks(np.arange(len(decoded)))
            ax.set_xticklabels(decoded, rotation=90)
            if answers is not None:
                ax.set_title(f'{answers[idx]}')
            ax.legend()
            prev_idx_pp=idx_pp

    # Hide any remaining empty subplots
    for ax in axes.flatten()[num_plots:]:
        ax.axis('off')
        
    plt.subplots_adjust(hspace=0.5, top=0.95)  
    plt.suptitle("Per-Word Perplexity across the Dataset", fontsize=30)
    plt.legend()
    plt.show()


from transformers import GPT2LMHeadModel, GPT2TokenizerFast

print("Importing model ...")
device = "cuda:3"
model_id = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

print(f"loaded model: {model_id}")
multisimo_df = pd.read_csv("/u/sebono/conversational_dominance/data/processed/transcript_dominance.csv")

ppl = {}
for el, path in zip(multisimo_df["file_content"], multisimo_df["file_name"]):
    print(f"processing file {path}")
    dataset = re.sub(r'\[', r'\n[', el).split("\n")[1:]
    pattern = r'\[(SPK\d|MOD)\]'
    matches = re.findall(pattern, "".join(dataset))
    speakers= np.unique(re.findall(pattern, "".join(dataset)))
    perpl = perplexity_of_fixedlength_models(dataset)
    ppl[path] = perpl


import pickle
file_path = "/u/sebono/conversational_dominance/notebooks/dominance_scores_multisimo.pkl"
with open(file_path, 'wb') as file:
    pickle.dump(ppl, file)