from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass
class Paths:
    """
    Dataclass for managing file paths based on account name.
    """

    account: str
    _PATH_TEMPLATES: ClassVar[dict[str, str]] = {
        "DATA_FILE": "{account}_tweets.json",
        "OUTPUT_CSV": "{account}_analysis.csv",
        "TEMPORAL_PLOT": "{account}_temporal_patterns.png",
        "NETWORK_PLOT": "{account}_network_graph.png",
        "WORDCLOUD_PLOT": "{account}_wordcloud.png",
    }
    DATA_FILE: Path = field(default_factory=lambda: Path(), init=False)
    OUTPUT_CSV: Path = field(default_factory=lambda: Path(), init=False)
    TEMPORAL_PLOT: Path = field(default_factory=lambda: Path(), init=False)
    NETWORK_PLOT: Path = field(default_factory=lambda: Path(), init=False)
    WORDCLOUD_PLOT: Path = field(default_factory=lambda: Path(), init=False)

    def __post_init__(self) -> None:
        """Initialize paths dynamically based on the account name."""
        self.__dict__.update(
            {
                attr: Path(template.format(account=self.account))
                for attr, template in self._PATH_TEMPLATES.items()
            }
        )
