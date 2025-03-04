import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from bot_analysis import TemporalAnalysisResult


def plot_temporal_patterns(
    temporal_data: TemporalAnalysisResult,
    path: Path,
    style: str = "seaborn-v0_8",
    dpi: int = 300,
) -> bool:
    """
    Plot temporal patterns with optimized plotting and robust error handling.

    Args:
        temporal_data: TemporalAnalysisResult object with temporal statistics.
        path: Path to save the plot.
        style: Matplotlib style context (default: "seaborn-v0_8").
        dpi: Resolution for saving the plot (default: 300).

    Returns:
        bool: True if plot saved successfully, False otherwise.

    """
    # Validate inputs via pydantic
    hour_counts = temporal_data.hour_counts
    day_counts = temporal_data.day_counts

    if hour_counts.empty and day_counts.empty:
        logging.warning("No temporal data to plot (both hour_counts and day_counts are empty).")
        return False

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        # Static day labels
        day_labels = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

        # Create figure with style context
        with plt.style.context(style):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
            ax1, ax2 = axes

            # Plot hourly plots if data exists
            if not hour_counts.empty:
                sns.barplot(
                    x=hour_counts.index,
                    y=hour_counts.to_numpy(),
                    ax=ax1,
                    palette="virdis",
                    hue=hour_counts.index,  # Avoid FutureWarning
                    legend=False,
                )
                ax1.set(
                    title="Tweets per Hour",
                    xlabel="Hour of Day",
                    ylabel="Number of Tweets",
                )
                ax1.tick_params(axis="x", rotation=45)
                ax1.grid(True, linestyle="--", alpha="0.7")

            # Plot daily counts if data exists
            if not day_counts.empty:
                sns.barplot(
                    x=day_counts.index,
                    y=day_counts.to_numpy(),
                    ax=ax2,
                    palette="virdis",
                    hue=day_counts.index,  # Avoid FutureWarning
                    legend=False,
                )
                ax2.set(
                    title="Tweets per Day of Week",
                    xlabel="Day of Week",
                    ylabel="Number of Tweets",
                )
                ax2.set_xticks(range(len(day_counts)))
                ax2.set_xticklabels([day_labels[i] for i in day_counts.index], rotation=45)
                ax2.grid(True, linestyle="--", alpha="0.7")

            # Save and close figure
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            logging.info("Temporal patterns plot saved to %s", path)
            return True
    except OSError:
        logging.exception("Failed to save plot to %s", path)
        return False
    except Exception:
        logging.exception("Error plotting temporal data")
        return False
