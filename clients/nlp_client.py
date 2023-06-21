from datatypes import gcp_nlp_types, aws_comprehend_types, nlp_client_types
from google.cloud import language_v1
from google.auth.credentials import Credentials
from botocore import session
from normalization.normalizer import NormalizeGcpSentiment, NormalizeAwsSentiment
from collections import defaultdict


def ConvertAwsComprehendResponse(response) -> aws_comprehend_types.Entities:
  entities = aws_comprehend_types.Entities(entities=[])

  for raw_entity in response.get('Entities', []):
    entity = aws_comprehend_types.Entity(text='', mentions=[])

    if 'DescriptiveMentionIndex' in raw_entity and len(
        raw_entity['DescriptiveMentionIndex']) > 0:
      # Able to find a descriptive mention index best matching this entity group.
      description_index = raw_entity['DescriptiveMentionIndex'][0]
      entity.text = raw_entity['Mentions'][description_index]['Text']

    for raw_mention in raw_entity['Mentions']:
      entity.mentions.append(
          aws_comprehend_types.Mention(
              text=raw_mention['Text'],
              score=raw_mention['Score'],
              group_score=raw_mention['GroupScore'],
              sentiments=aws_comprehend_types.SentimentScore(
                  positive=raw_mention['MentionSentiment']['SentimentScore']
                  ['Positive'],
                  negative=raw_mention['MentionSentiment']['SentimentScore']
                  ['Negative'])))

    entities.entities.append(entity)

  return entities


def ConvertGcpNlpResponse(
    response: language_v1.AnalyzeEntitySentimentResponse
) -> gcp_nlp_types.Entities:
  gcp_entities = gcp_nlp_types.Entities(entities=[])
  for raw_entity in response.entities:
    gcp_entities.entities.append(
        gcp_nlp_types.Entity(name=raw_entity.name,
                             salience=raw_entity.salience,
                             sentiment=gcp_nlp_types.Sentiment(
                                 score=raw_entity.sentiment.score,
                                 magnitude=raw_entity.sentiment.magnitude)))
  return gcp_entities


def ComputeSentimentInMergedEntity(
    entity: nlp_client_types.MergedNlpEntity) -> nlp_client_types.Sentiment:
  ''' Returns a unanimous sentiment derived from ALL results of NLP sentiment analysis to the specified entity. 
  
    Positive: When all sentiments are positive.
    Negative: When all sentiments are negative.
    Neutral: When all sentiments exhibit a state of neutrality within a defined range.
    Unsure: Cannot decide. 
  '''
  scores = [entity.aws_score, entity.gcp_score]

  if all(abs(score) <= 0.1 for score in scores):
    return nlp_client_types.Sentiment.Neutral
  elif all(score > 0 for score in scores):
    return nlp_client_types.Sentiment.Positive
  elif all(score < 0 for score in scores):
    return nlp_client_types.Sentiment.Negative

  return nlp_client_types.Sentiment.Unsure


def MergeEntities(
    aws_entities: nlp_client_types.NlpEntities,
    gcp_entities: nlp_client_types.NlpEntities
) -> nlp_client_types.MergedNlpEntities:

  text_to_entitiy_sentiment = defaultdict(
      lambda: nlp_client_types.MergedNlpEntity(
          text=None, aws_score=None, gcp_score=None, overall_sentiment=None))

  for aws_entity in aws_entities.entities:
    text_to_entitiy_sentiment[aws_entity.text].text = aws_entity.text
    text_to_entitiy_sentiment[
        aws_entity.text].aws_score = aws_entity.sentiment_score
  for gcp_entity in gcp_entities.entities:
    text_to_entitiy_sentiment[gcp_entity.text].text = gcp_entity.text
    text_to_entitiy_sentiment[
        gcp_entity.text].gcp_score = gcp_entity.sentiment_score

  merged_nlp_entities = nlp_client_types.MergedNlpEntities(
      common_entities=[],
      entities=[
          text_to_entitiy_sentiment[text]
          for text in sorted(text_to_entitiy_sentiment)
      ])

  # Find common entities occured in ALL NLP analysis tools.
  merged_nlp_entities.common_entities = [
      entity for entity in merged_nlp_entities.entities
      if entity.gcp_score is not None and entity.aws_score is not None
  ]

  return merged_nlp_entities


class NlpClient:

  def __init__(self, aws_comprehend_client=None, gcp_nlp_client=None):
    self.aws_comprehend_client = aws_comprehend_client
    self.gcp_nlp_client = gcp_nlp_client

  @classmethod
  def NewNlpClient(cls, aws_credentials, gcp_credentials: Credentials):
    return NlpClient(
        aws_comprehend_client=session.Session().create_client(
            'comprehend',
            region_name='us-west-2',
            aws_access_key_id=aws_credentials.access_key_id,
            aws_secret_access_key=aws_credentials.secret_access_key),
        gcp_nlp_client=language_v1.LanguageServiceClient(
            credentials=gcp_credentials))

  def AnalyzeSentiment(self, text: str) -> nlp_client_types.MergedNlpEntities:
    aws_response = self.aws_comprehend_client.detect_targeted_sentiment(
        Text=text  # UTF-8 encoded text, maximum string size 5KB.
        ,
        LanguageCode='en'  # English (en) is the only supported language.
    )
    aws_entities = NormalizeAwsSentiment(
        ConvertAwsComprehendResponse(aws_response))

    gcp_response = self.gcp_nlp_client.analyze_entity_sentiment(
        request={
            "document": {
                "content": text,
                "type_": language_v1.types.Document.Type.PLAIN_TEXT,
                # "language": "en" # Optional. If not specified, the language is automatically detected.
            },
            "encoding_type": language_v1.EncodingType.UTF8
        })
    gcp_entities = NormalizeGcpSentiment(ConvertGcpNlpResponse(gcp_response))

    merged_entities = MergeEntities(aws_entities, gcp_entities)

    # Label sentiment to common entities.
    for entity in merged_entities.common_entities:
      entity.overall_sentiment = ComputeSentimentInMergedEntity(entity)

    return merged_entities