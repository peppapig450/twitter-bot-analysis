import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from wordcloud import WordCloud

from bot_analysis import ContentAnalysisResult


def plot_content_visualizations(
    content_data: ContentAnalysisResult,
    path: Path,
    style: str = "seaborn-v0_8",
    dpi: int = 300,
    wordcloud_params: dict[str, Any] | None = None,
) -> bool:
    """
    Generate and save a word cloud for top terms in content analysis.

    Args:
        content_data: ContentAnalysisResult object with top terms.
        path: Path to save the word cloud image.
        style: Matplotlib style context (default: "seaborn-v0_8").
        dpi: Resolution for saving the plot (default: 300).
        wordcloud_params: Optional dict of additional WordCloud parameters.

    Returns:
        bool: True if plot saved successfully, False otherwise.

    """
    top_terms = content_data.top_terms
    if not top_terms:
        logging.warning("No top terms provided for word cloud plotting.")
        return False

    try:
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Default WordCloud parameters with option to override
        default_params: dict[str, Any] = {
            "width": 800,
            "height": 400,
            "background_color": "white",
            "max_words": 50,
        }
        if wordcloud_params:
            default_params.update(wordcloud_params)

        # Validate top_terms are strings
        if not all(isinstance(term, str) for term in top_terms):
            error_message = "All top_terms must be strings."
            raise ValueError(error_message)

        # Generate word cloud
        wordcloud = WordCloud(**default_params).generate(" ".join(top_terms))

        # Plot with style context
        with plt.style.context(style):
            fig = plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title("Word Cloud of Top Terms", pad=10)  # Add padding to avoid overlap
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        logging.info("Word cloud plot saved to %s", path)

    except OSError:
        logging.exception("Failed to save word cloud to %s", path)
        return False
    except ValueError:
        logging.exception("Invalid data for word cloud")
        return False
    except Exception:
        logging.exception("Error generating word cloud: %s")
        return False
    else:
        return True
