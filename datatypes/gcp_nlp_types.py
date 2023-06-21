'''
  Data types for GCP Natural Language API.
  Reference on raw response https://cloud.google.com/natural-language/docs/basics#entity-sentiment-analysis.
'''

from dataclasses import dataclass
from typing import List


@dataclass
class Sentiment:
  score: float  # [-1 (negative), 1 (positive)].
  magnitude: float  # [0 (week), inf (strong)). Sentiment strength.


@dataclass
class Entity:
  name: str
  salience: float  # [0 (less relevant), 1 (highly relevent)]. The relevance of the entity to the entire text.
  sentiment: Sentiment  # The aggregated sentiment of all the mentions of this entity.


@dataclass
class Entities:
  entities: List[Entity]