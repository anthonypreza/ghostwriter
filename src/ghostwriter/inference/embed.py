import torch
import torch.nn.functional as F


@torch.inference_mode()
def embed_song(model, tokenizer, lyrics, context_length=128, device_name="cuda"):
    model.eval()
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)

    ids = tokenizer.encode(lyrics)

    total = torch.zeros(model.embedding.embedding_dim, device=device)
    total_tokens = 0

    for start in range(0, len(ids), context_length):
        chunk = ids[start : start + context_length]

        x = torch.tensor(
            [chunk],
            dtype=torch.long,
            device=device,
        )

        # (1, T, D)
        hidden = model(x, return_hidden=True)

        # Average token embeddings into one window embedding
        # (1, D)
        window_embedding = hidden.mean(dim=1)

        # Weight by # of tokens in this window
        total += window_embedding.squeeze(0) * len(chunk)
        total_tokens += len(chunk)

    # Average across all windows
    song_embedding = total / total_tokens

    return F.normalize(song_embedding, dim=0).cpu().numpy()
