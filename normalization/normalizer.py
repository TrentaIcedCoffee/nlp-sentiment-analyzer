from datatypes import aws_types, gcp_types, nlp_client_types
import collections


def ArithmeticMean(values):
  return sum(values) / len(values)


def NormalizeGcpSentiment(
    gcp_entities: list[gcp_types.GcpEntity]) -> list[nlp_client_types.Entity]:

  def NormalizeEach(
      text: str,
      gcp_entities: list[gcp_types.GcpEntity]) -> nlp_client_types.Entity:
    ''' Normalize each entity group with the same text. '''
    weighted_sentiments = []
    for entity in gcp_entities:
      # Normalize magnitudes from [0, inf) to [0, 1)
      normalized_magnitude = entity.sentiment.magnitude / (
          entity.sentiment.magnitude + 1)
      weighted_sentiments.append(entity.sentiment.score * normalized_magnitude)
    return nlp_client_types.Entity(
        text=text, gcp_score=ArithmeticMean(weighted_sentiments))

  entities = []

  # Reduce entities by entity name (entity text).
  text_to_entities = collections.defaultdict(list)
  for gcp_entity in gcp_entities:
    text_to_entities[gcp_entity.name].append(gcp_entity)
  for text, gcp_entities in text_to_entities.items():
    entities.append(NormalizeEach(text, gcp_entities))

  return entities


def NormalizeAwsSentiment(
    aws_entities: list[aws_types.AwsEntity]) -> list[nlp_client_types.Entity]:

  def NormalizeEach(
      text: str,
      aws_entities: list[aws_types.AwsEntity]) -> nlp_client_types.Entity:
    ''' Normalize each entity group with the same text. '''
    weighted_sentiments = []
    for aws_entity in aws_entities:
      for mention in aws_entity.mentions:
        if mention.group_score < 0.5:
          continue  # Skip mentions with low group score, which are likely not related to the same entity.
        weighted_sentiments.append(
            (mention.sentiments.positive - mention.sentiments.negative) *
            mention.score)
    return nlp_client_types.Entity(
        text=text, aws_score=ArithmeticMean(weighted_sentiments))

  entities = []

  # Reduce entities by entity name (entity text).
  text_to_entities = collections.defaultdict(list)
  for aws_entity in aws_entities:
    text_to_entities[aws_entity.text].append(aws_entity)
  for text, aws_entities in text_to_entities.items():
    entities.append(NormalizeEach(text, aws_entities))

  return entities