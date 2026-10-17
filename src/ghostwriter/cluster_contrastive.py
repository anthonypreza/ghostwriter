"""Compare pooled and projected lyric embeddings with K-means and UMAP."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F
import umap
from sklearn.cluster import KMeans

from .training.contrastive_common import load_contrastive


@torch.inference_mode()
def embed_song_views(model, tokenizer, projection, lyrics, device, context_length):
    """Average causal token states across a whole song, then apply the head."""
    ids = [token for token in tokenizer.encode(lyrics) if token not in (2, 3)]
    if not ids:
        raise ValueError("A song has no usable tokens.")
    model.eval()
    projection.eval()
    total = torch.zeros(model.embedding.embedding_dim, device=device)
    for start in range(0, len(ids), context_length):
        chunk = ids[start : start + context_length]
        inputs = torch.tensor([chunk], dtype=torch.long, device=device)
        total += model(inputs, return_hidden=True).sum(dim=1).squeeze(0)
    pooled = total / len(ids)
    original = F.normalize(pooled, dim=0)
    projected = F.normalize(projection(pooled.unsqueeze(0)), dim=1).squeeze(0)
    return original.cpu().numpy(), projected.cpu().numpy()


def artist_precision_at_five(embeddings, artists):
    if len(artists) < 6:
        return None
    similarities = embeddings @ embeddings.T
    np.fill_diagonal(similarities, -np.inf)
    neighbors = np.argsort(similarities, axis=1)[:, -5:]
    labels = np.asarray(artists)
    return float((labels[neighbors] == labels[:, None]).mean())


def visualize(
    embeddings, names, splits, representation, output_path, n_clusters, show=False
):
    if len(names) < 3:
        raise ValueError("Visualization needs at least three songs.")
    artists = [name.split("__", 1)[0] if "__" in name else "unknown" for name in names]
    clusters = KMeans(
        n_clusters=min(n_clusters, len(names)), random_state=42, n_init=10
    ).fit_predict(embeddings)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, len(names) - 1),
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)
    frame = pd.DataFrame(
        {
            "song": names,
            "artist": artists,
            "split": splits,
            "cluster": clusters.astype(str),
            "x": coords[:, 0],
            "y": coords[:, 1],
        }
    )
    fig = px.scatter(
        frame,
        x="x",
        y="y",
        color="cluster",
        hover_name="song",
        hover_data={"artist": True, "split": True, "x": False, "y": False},
        title=f"Ghostwriter | {representation.capitalize()} song embeddings",
        labels={"x": "UMAP 1", "y": "UMAP 2", "cluster": "Cluster"},
        opacity=0.75,
        template="plotly_dark",
    )
    fig.update_traces(marker={"size": 8})
    fig.update_layout(width=1200, height=800, hovermode="closest")
    fig.write_html(str(output_path), include_plotlyjs=True)
    if show:
        fig.show()
    precision = artist_precision_at_five(embeddings, artists)
    print(
        f"{representation.capitalize()} | Artist precision@5: {precision:.1%}"
        if precision is not None
        else representation
    )
    for cluster_id in sorted(set(clusters)):
        indices = np.where(clusters == cluster_id)[0]
        top_artists = pd.Series(np.asarray(artists)[indices]).value_counts().head(3)
        summary = ", ".join(
            f"{artist} {count}" for artist, count in top_artists.items()
        )
        print(f"  Cluster {cluster_id}: {len(indices)} songs | {summary}")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contrastive-dir", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "val", "test", "all"), default="test"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument(
        "--show", action="store_true", help="Also open the charts in a browser."
    )
    args = parser.parse_args()
    if args.clusters < 1:
        parser.error("--clusters must be positive.")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, projection, config, _ = load_contrastive(
        args.contrastive_dir, device
    )
    splits_to_plot = ("train", "val", "test") if args.split == "all" else (args.split,)
    paths = [
        (split, path)
        for split in splits_to_plot
        for path in sorted(
            (Path(config["base_run_dir"]) / "processed" / split).glob("*.txt")
        )
    ]
    if not paths:
        raise ValueError("No processed songs found for the requested split.")
    names, splits, original, projected = [], [], [], []
    for split, path in paths:
        lyrics = path.read_text(encoding="utf-8")
        if not any(token not in (2, 3) for token in tokenizer.encode(lyrics)):
            continue
        base_vector, projected_vector = embed_song_views(
            model, tokenizer, projection, lyrics, device, config["context_length"]
        )
        names.append(path.stem)
        splits.append(split)
        original.append(base_vector)
        projected.append(projected_vector)
    print(f"Visualizing {len(names)} songs from {', '.join(splits_to_plot)}.")
    for representation, vectors in (("pooled", original), ("projected", projected)):
        output_path = (
            Path(args.contrastive_dir) / f"clusters_{args.split}_{representation}.html"
        )
        visualize(
            np.stack(vectors),
            names,
            splits,
            representation,
            output_path,
            args.clusters,
            args.show,
        )
    print(
        "UMAP is fitted separately for each view; compare neighborhoods, not axis positions."
    )


if __name__ == "__main__":
    main()
