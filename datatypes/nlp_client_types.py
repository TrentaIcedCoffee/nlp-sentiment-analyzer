from typing import Optional

import enum
import dataclasses
import json


class Sentiment(enum.Enum):
  Unsure = 0  # inadequate confidence or certainty.
  Positive = 1
  Negative = 2
  Neutral = 3  # Sentiment is neutral or mixed.


@dataclasses.dataclass
class Entity:
  text: str
  aws_score: Optional[float] = None  # [-1 (negative), 1 (positive)]
  gcp_score: Optional[float] = None  # [-1 (negative), 1 (positive)]
  overall_sentiment: Optional[Sentiment] = None

  def to_dict(self) -> dict:
    obj = dataclasses.asdict(self)
    obj['overall_sentiment'] = self.overall_sentiment.name if self.overall_sentiment else None

    return obj


@dataclasses.dataclass
class MergedNlpEntities:
  common_entities: list[Entity]  # Entities occured in ALL NLP analysis tools.
  entities: list[Entity]  # Entities occured in ANY NLP analysis tool.

  def to_dict(self) -> dict:
    return {
        'common_entities': [
            entity.to_dict() for entity in self.common_entities
        ],
        'entities': [entity.to_dict() for entity in self.entities]
    }

  def to_json(self) -> str:
    return json.dumps(self.to_dict(), indent=2, sort_keys=True)
