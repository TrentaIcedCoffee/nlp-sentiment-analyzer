from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Optional


class Sentiment(Enum):
  Unsure = 0  # inadequate confidence or certainty.
  Positive = 1
  Negative = 2
  Neutral = 3  # Sentiment is neutral or mixed.

  def ToJson(self):
    return self.name


@dataclass
class NlpEntity:
  text: str
  sentiment_score: float  # [-1 (negative), 1 (positive)]

  def ToJson(self):
    return asdict(self)


@dataclass
class NlpEntities:
  entities: List[NlpEntity]

  def ToJson(self):
    return {'entities': [entity.ToJson() for entity in self.entities]}


# NLP entity with sentiments from all NLP analysis tools.
@dataclass
class MergedNlpEntity:
  text: str
  aws_score: Optional[float]  # [-1 (negative), 1 (positive)]
  gcp_score: Optional[float]  # [-1 (negative), 1 (positive)]
  overall_sentiment: Optional[Sentiment]

  def ToJson(self):
    return {
        **asdict(self), 'overall_sentiment':
            self.overall_sentiment.ToJson()
            if self.overall_sentiment is not None else None
    }


@dataclass
class MergedNlpEntities:
  common_entities: List[
      MergedNlpEntity]  # Entities occured in ALL NLP analysis tools.
  entities: List[MergedNlpEntity]  # Entities occured in ANY NLP analysis tool.

  def ToJson(self):
    return {
        'common_entities': [entity.ToJson() for entity in self.common_entities],
        'entities': [entity.ToJson() for entity in self.entities]
    }
