import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import gradio as gr
import tempfile

def water_leakage_analysis(num_days=30, z_threshold=2.0):
    df = pd.DataFrame({
        "day": range(1, num_days + 1),
        "water_supplied": np.random.randint(1000, 1500, num_days)
    })

    df["water_consumed"] = df["water_supplied"] - np.random.randint(50, 300, num_days)
    df["water_loss"] = df["water_supplied"] - df["water_consumed"]

    df["z_score"] = zscore(df["water_loss"])
    df["leakage_detected"] = df["z_score"].abs() > z_threshold

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["day"], df["water_loss"], label="Water Loss")
    ax.scatter(
        df[df["leakage_detected"]]["day"],
        df[df["leakage_detected"]]["water_loss"],
        color="red",
        label="Leakage"
    )
    ax.set_xlabel("Day")
    ax.set_ylabel("Water Loss")
    ax.set_title("City Water Pipeline Leakage Detection")
    ax.legend()
    ax.grid(True)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(tmp.name)
    plt.close(fig)

    return df, tmp.name


app = gr.Interface(
    fn=water_leakage_analysis,
    inputs=[
        gr.Slider(10, 60, value=30, step=5, label="Number of Days"),
        gr.Slider(1.0, 3.0, value=2.0, step=0.1, label="Z-Score Threshold")
    ],
    outputs=[
        gr.Dataframe(label="Water Pipeline Data"),
        gr.Image(label="Leakage Visualization")
    ],
    title="🚰 City Water Pipeline Leakage Analysis",
    description="Permanent web-based AIML application using statistical anomaly detection"
)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        debug=False,
        show_error=False
    )


