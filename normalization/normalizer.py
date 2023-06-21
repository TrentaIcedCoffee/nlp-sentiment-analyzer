from datatypes import gcp_nlp_types, aws_comprehend_types, nlp_client_types
from collections import defaultdict
from typing import List


def ArithmeticMean(values):
  return sum(values) / len(values)


def NormalizeGcpSentiment(
    entities: gcp_nlp_types.Entities) -> nlp_client_types.NlpEntities:

  def NormalizeEach(
      text: str,
      entities: List[gcp_nlp_types.Entity]) -> nlp_client_types.NlpEntity:
    ''' Normalize each entity group with the same text. '''
    weighted_sentiments = []
    for entity in entities:
      # Normalize magnitudes from [0, inf) to [0, 1)
      normalized_magnitude = entity.sentiment.magnitude / (
          entity.sentiment.magnitude + 1)
      weighted_sentiments.append(entity.sentiment.score * normalized_magnitude)
    aggregated_sentiment = ArithmeticMean(weighted_sentiments)
    return nlp_client_types.NlpEntity(
        text=text, sentiment_score=aggregated_sentiment)

  result = nlp_client_types.NlpEntities(entities=[])

  # Reduce entities by entity name (entity text).
  text_to_entities = defaultdict(list)
  for entity in entities.entities:
    text_to_entities[entity.name].append(entity)
  for text, entities in text_to_entities.items():
    result.entities.append(NormalizeEach(text, entities))

  return result


def NormalizeAwsSentiment(
    entities: aws_comprehend_types.Entities
) -> nlp_client_types.NlpEntities:

  def NormalizeEach(
      text: str, entities: List[aws_comprehend_types.Entity]
  ) -> nlp_client_types.NlpEntity:
    ''' Normalize each entity group with the same text. '''
    weighted_sentiments = []
    for entity in entities:
      for mention in entity.mentions:
        if mention.group_score < 0.5:
          continue  # Skip mentions with low group score, which are likely not related to the same entity.
        weighted_sentiments.append(
            (mention.sentiments.positive - mention.sentiments.negative) *
            mention.score)
    aggregated_sentiment = ArithmeticMean(weighted_sentiments)
    return nlp_client_types.NlpEntity(
        text=text, sentiment_score=aggregated_sentiment)

  result = nlp_client_types.NlpEntities(entities=[])

  # Reduce entities by entity name (entity text).
  text_to_entities = defaultdict(list)
  for entity in entities.entities:
    text_to_entities[entity.text].append(entity)
  for text, entities in text_to_entities.items():
    result.entities.append(NormalizeEach(text, entities))

  return result