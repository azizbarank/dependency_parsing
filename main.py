import sys
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm
import networkx as nx


class DependencyParser(nn.Module):
    def __init__(self, encoder_name="roberta-base", mlp_dim=500):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.hidden_size = self.encoder.config.hidden_size  # 768 for roberta-base
        self.mlp_dim = mlp_dim

        # MLP for head representations
        self.mlp_head = nn.Sequential(
            nn.Linear(self.hidden_size, mlp_dim),
            nn.ReLU()
        )

        # MLP for dependent representations
        self.mlp_dep = nn.Sequential(
            nn.Linear(self.hidden_size, mlp_dim),
            nn.ReLU()
        )

        # Biaffine attention parameters
        self.U1 = nn.Parameter(torch.empty(mlp_dim, mlp_dim))
        self.u2 = nn.Parameter(torch.empty(mlp_dim))
        nn.init.xavier_uniform_(self.U1)
        nn.init.zeros_(self.u2)

    def forward(self, input_ids, attention_mask, head=None):
        # Step 5: Get RoBERTa embeddings
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Step 6: Apply MLPs to get H_head and H_dep
        H_head = self.mlp_head(hidden_states)  # (batch, seq_len, mlp_dim)
        H_dep = self.mlp_dep(hidden_states)    # (batch, seq_len, mlp_dim)

        # Step 7: Compute biaffine scores (vectorized)
        # score(i,j) = H_head[i].T @ U1 @ H_dep[j] + H_head[i].T @ u2
        bilinear = torch.einsum('bid,de,bje->bij', H_head, self.U1, H_dep)
        linear = torch.einsum('bid,d->bi', H_head, self.u2)
        scores = bilinear + linear.unsqueeze(2)  # (batch, seq_len, seq_len)

        # Step 8: Compute loss if labels provided
        loss = None
        if head is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(scores, head)

        return scores, loss

tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
dataset = load_dataset("universal_dependencies", "en_ewt", trust_remote_code=True)

all_deprels = set()
for example in dataset["train"]:
    for deprel in example["deprel"]:
        if deprel != "None":
            all_deprels.add(deprel)

deprel_to_id = {deprel: i for i, deprel in enumerate(sorted(all_deprels))}
id_to_deprel = {i: deprel for deprel, i in deprel_to_id.items()}


def strip_none_heads(examples, i):
    tokens = examples["tokens"][i]
    heads = examples["head"][i]
    deprels = examples["deprel"][i]
    non_none = [(t, h, d) for t, h, d in zip(tokens, heads, deprels) if h != "None"]
    return zip(*non_none)


def map_first_occurrence(nums):
    seen = set()
    return {num: i for i, num in enumerate(nums) if num is not None and num not in seen and not seen.add(num)}


def tokenize_and_align_labels(examples, skip_index=-100):
    examples_tokens, examples_heads, examples_deprels = [], [], []
    for sentence_id in range(len(examples["tokens"])):
        tt, hh, dd = strip_none_heads(examples, sentence_id)
        examples_tokens.append(tt)
        examples_heads.append(hh)
        examples_deprels.append(dd)

    tokenized_inputs = tokenizer(examples_tokens, truncation=True, is_split_into_words=True, padding=True)

    remapped_heads = []
    deprel_ids = []
    tokens_representing_words = []
    num_words = []
    maxlen_t2w = 0

    for sentence_id, annotated_heads in enumerate(examples_heads):
        deprels = examples_deprels[sentence_id]
        word_ids = tokenized_inputs.word_ids(batch_index=sentence_id)
        word_pos_to_token_pos = map_first_occurrence(word_ids)

        previous_word_idx = None
        heads_here = []
        deprel_ids_here = []
        tokens_representing_word_here = [0]

        for sentence_position, word_idx in enumerate(word_ids):
            if word_idx is None:
                heads_here.append(skip_index)
                deprel_ids_here.append(skip_index)
            elif word_idx != previous_word_idx:
                if annotated_heads[word_idx] == "None":
                    print("A 'None' head survived!")
                    sys.exit(0)
                else:
                    head_word_pos = int(annotated_heads[word_idx])
                    head_token_pos = 0 if head_word_pos == 0 else word_pos_to_token_pos[head_word_pos - 1]
                    heads_here.append(head_token_pos)
                    deprel_ids_here.append(deprel_to_id[deprels[word_idx]])
                    tokens_representing_word_here.append(sentence_position)
            else:
                heads_here.append(skip_index)
                deprel_ids_here.append(skip_index)
            previous_word_idx = word_idx

        remapped_heads.append(heads_here)
        deprel_ids.append(deprel_ids_here)
        tokens_representing_words.append(tokens_representing_word_here)
        num_words.append(len(tokens_representing_word_here))
        if len(tokens_representing_word_here) > maxlen_t2w:
            maxlen_t2w = len(tokens_representing_word_here)

    for t2w in tokens_representing_words:
        t2w += [-1] * (maxlen_t2w - len(t2w))

    tokenized_inputs["head"] = remapped_heads
    tokenized_inputs["deprel_ids"] = deprel_ids
    tokenized_inputs["tokens_representing_words"] = tokens_representing_words
    tokenized_inputs["num_words"] = num_words
    tokenized_inputs["tokenid_to_wordid"] = [tokenized_inputs.word_ids(batch_index=i) for i in range(len(examples_heads))]

    return tokenized_inputs


def visualize_sentence(tokenized_data, idx):
    input_ids = tokenized_data["input_ids"][idx]
    heads = tokenized_data["head"][idx]
    deprel_ids = tokenized_data["deprel_ids"][idx]
    word_ids = tokenized_data["tokenid_to_wordid"][idx]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    print(f"{'Pos':<5} {'Token':<20} {'WordID':<8} {'Head':<6} {'DepRel':<15}")
    print("-" * 60)
    for pos, (token, head, deprel_id, word_id) in enumerate(zip(tokens, heads, deprel_ids, word_ids)):
        deprel = id_to_deprel[deprel_id] if deprel_id != -100 else "-"
        head_str = str(head) if head != -100 else "-"
        word_id_str = str(word_id) if word_id is not None else "-"
        print(f"{pos:<5} {token:<20} {word_id_str:<8} {head_str:<6} {deprel:<15}")
    print()


def collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    max_words = max(len(x["tokens_representing_words"]) for x in batch)

    def pad(seq, length, value=0):
        return seq + [value] * (length - len(seq))

    return {
        "input_ids": torch.tensor([pad(x["input_ids"], max_len, tokenizer.pad_token_id) for x in batch]),
        "attention_mask": torch.tensor([pad(x["attention_mask"], max_len, 0) for x in batch]),
        "head": torch.tensor([pad(x["head"], max_len, -100) for x in batch]),
        "deprel_ids": torch.tensor([pad(x["deprel_ids"], max_len, -100) for x in batch]),
        "tokens_representing_words": torch.tensor([pad(x["tokens_representing_words"], max_words, -1) for x in batch]),
        "num_words": torch.tensor([x["num_words"] for x in batch]),
    }


def compute_head_accuracy(model, dataloader, device):
    """Compute head tagging accuracy, ignoring tokens with head=-100."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            heads = batch["head"].to(device)

            scores, _ = model(input_ids, attention_mask)
            predictions = scores.argmax(dim=1)  # (batch, seq_len)

            mask = heads != -100
            correct += (predictions[mask] == heads[mask]).sum().item()
            total += mask.sum().item()

    model.train()
    return correct / total if total > 0 else 0.0


def mst_decode(scores, word_pos, n_words):
    """Decode best tree using Chu-Liu-Edmonds. Returns list of head word indices."""
    G = nx.DiGraph()
    for dep in range(1, n_words):
        for head in range(n_words):
            if head != dep:
                G.add_edge(head, dep, weight=scores[word_pos[head], word_pos[dep]].item())

    try:
        mst = nx.maximum_spanning_arborescence(G, attr='weight')
        heads = {dep: head for head, dep in mst.edges()}
        return [heads.get(i, 0) for i in range(n_words)]
    except nx.NetworkXException:
        return [scores[word_pos[:n_words], word_pos[i]].argmax().item() for i in range(n_words)]


def compute_uas(model, dataloader, device):
    """Compute UAS using MST decoding."""
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing UAS"):
            scores, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            log_probs = F.log_softmax(scores, dim=1).cpu()

            for b in range(scores.size(0)):
                n = batch["num_words"][b].item()
                wp = batch["tokens_representing_words"][b][:n].tolist()
                preds = mst_decode(log_probs[b], wp, n)

                for w in range(1, n):
                    gold_head_tok = batch["head"][b, wp[w]].item()
                    gold_head_word = next((i for i, p in enumerate(wp) if p == gold_head_tok), 0)
                    correct += (preds[w] == gold_head_word)
                    total += 1

    model.train()
    return correct / total if total > 0 else 0.0


def train(model, train_loader, dev_loader, device, num_epochs=10, lr=2e-5):
    """Training loop with wandb logging and head accuracy evaluation."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            heads = batch["head"].to(device)

            optimizer.zero_grad()
            scores, loss = model(input_ids, attention_mask, heads)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        head_acc = compute_head_accuracy(model, dev_loader, device)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "head_accuracy": head_acc
        })

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Head Accuracy={head_acc:.4f}")

    return model


def main():
    print(f"Deprels: {len(deprel_to_id)}, Train: {len(dataset['train'])}, Dev: {len(dataset['validation'])}")

    train_tokenized = dataset["train"].map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    dev_tokenized = dataset["validation"].map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )

    train_loader = DataLoader(train_tokenized, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_tokenized, batch_size=32, shuffle=False, collate_fn=collate_fn)

    print(f"Train batches: {len(train_loader)}, Dev batches: {len(dev_loader)}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = DependencyParser()

    # Initialize wandb (offline mode - no login required)
    wandb.init(project="dependency-parsing", mode="offline", config={
        "learning_rate": 2e-5,
        "epochs": 10,
        "batch_size": 32,
        "mlp_dim": 500,
        "encoder": "roberta-base"
    })

    # Train
    model = train(model, train_loader, dev_loader, device, num_epochs=10, lr=2e-5)

    # Final evaluation
    final_acc = compute_head_accuracy(model, dev_loader, device)
    print(f"\nFinal Head Tagging Accuracy: {final_acc:.4f}")

    # UAS with MST decoding
    uas = compute_uas(model, dev_loader, device)
    print(f"UAS (MST): {uas:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()
