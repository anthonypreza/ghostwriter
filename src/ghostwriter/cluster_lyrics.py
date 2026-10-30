import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import umap
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from .data.prepare import PROCESSED_DIR
from .inference.embed import embed_song
from .training.checkpoint import (
    DEFAULT_CHECKPOINT,
    DEFAULT_TOKENIZER,
    load_model,
)


def main():
    parser = argparse.ArgumentParser(description="Visualize song embeddings")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "outputs" / "visualizations",
        help="Directory for the generated interactive chart.",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(
        args.checkpoint,
        args.tokenizer,
        args.device,
    )

    song_dirs = [
        args.data_dir / "train",
        args.data_dir / "val",
        args.data_dir / "test",
    ]

    songs = []
    filenames = []
    splits = []

    for directory in song_dirs:
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue

            with path.open("r", encoding="utf-8") as f:
                songs.append(f.read())

            filenames.append(path.stem)
            splits.append(directory.name)

    embeddings = np.stack([
        embed_song(
            model,
            tokenizer,
            song,
            context_length=model.context_length or 128,
            device_name=args.device,
        )
        for song in songs
    ])

    # Cluster in the original 64-dimensional space.
    kmeans = KMeans(
        n_clusters=8,
        random_state=42,
        n_init=10,
    )
    clusters = kmeans.fit_predict(embeddings)

    for cluster_id in range(8):
        indices = np.where(clusters == cluster_id)[0]

        centroid = embeddings[indices].mean(axis=0, keepdims=True)

        similarities = cosine_similarity(
            embeddings[indices],
            centroid,
        ).ravel()

        representative_indices = indices[np.argsort(similarities)[::-1][:10]]

        print(f"\nCLUSTER {cluster_id} ({len(indices)} songs)")

        for idx in representative_indices:
            print(f"  {filenames[idx]}")

    # Project to 2D for visualization.
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)

    # Build interactive visualization.
    df = pd.DataFrame(
        {
            "song": filenames,
            "split": splits,
            "cluster": clusters.astype(str),
            "x": coords[:, 0],
            "y": coords[:, 1],
        }
    )

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster",
        hover_name="song",
        hover_data={
            "split": True,
            "cluster": True,
            "x": False,
            "y": False,
        },
        title="Ghostwriter: Song Embedding Clusters",
        labels={
            "x": "UMAP 1",
            "y": "UMAP 2",
            "cluster": "Cluster",
        },
        opacity=0.75,
        template="plotly_dark",
    )

    fig.update_traces(
        marker={"size": 8},
    )

    fig.update_layout(
        width=1200,
        height=800,
        hovermode="closest",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "ghostwriter_clusters.html"
    fig.write_html(str(output_path), include_plotlyjs=True)
    fig.show()

    print(f"Saved interactive visualization to {output_path.resolve()}")


if __name__ == "__main__":
    main()
