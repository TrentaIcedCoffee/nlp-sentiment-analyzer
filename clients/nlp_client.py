from datatypes import aws_types, gcp_types, nlp_client_types
from google.cloud import language_v1
from google.auth import credentials
from botocore import session
from normalization import normalizer
import collections


def convert_aws_response(response) -> list[aws_types.AwsEntity]:
  aws_entities = []

  for raw_entity in response.get('Entities', []):
    aws_entity = aws_types.AwsEntity(text='', mentions=[])

    if raw_entity.get('DescriptiveMentionIndex'):
      # Able to find a descriptive mention index best matching this entity group.
      description_index = raw_entity['DescriptiveMentionIndex'][0]
      aws_entity.text = raw_entity['Mentions'][description_index]['Text']

    for raw_mention in raw_entity['Mentions']:
      aws_entity.mentions.append(
          aws_types.Mention(text=raw_mention['Text'],
                            score=raw_mention['Score'],
                            group_score=raw_mention['GroupScore'],
                            sentiments=aws_types.SentimentScore(
                                positive=raw_mention['MentionSentiment']
                                ['SentimentScore']['Positive'],
                                negative=raw_mention['MentionSentiment']
                                ['SentimentScore']['Negative'])))

    aws_entities.append(aws_entity)

  return aws_entities


def convert_gcp_response(
    response: language_v1.AnalyzeEntitySentimentResponse
) -> list[gcp_types.GcpEntity]:
  return list(
      map(
          lambda raw_entity: gcp_types.GcpEntity(
              name=raw_entity.name,
              salience=raw_entity.salience,
              sentiment=gcp_types.Sentiment(score=raw_entity.sentiment.score,
                                            magnitude=raw_entity.sentiment.
                                            magnitude)),
          response.entities,
      ))


def ComputeSentimentInMergedEntity(
    entity: nlp_client_types.Entity) -> nlp_client_types.Sentiment:
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


# TODO: refactor here.
def MergeEntities(
    aws_entities: list[nlp_client_types.Entity],
    gcp_entities: list[nlp_client_types.Entity]
) -> list[nlp_client_types.Entity]:

  text_to_entitiy_sentiment = collections.defaultdict(
      lambda: nlp_client_types.Entity(
          text=None, aws_score=None, gcp_score=None, overall_sentiment=None))

  for aws_entity in aws_entities:
    text_to_entitiy_sentiment[aws_entity.text].text = aws_entity.text
    text_to_entitiy_sentiment[aws_entity.text].aws_score = aws_entity.aws_score
  for gcp_entity in gcp_entities:
    text_to_entitiy_sentiment[gcp_entity.text].text = gcp_entity.text
    text_to_entitiy_sentiment[gcp_entity.text].gcp_score = gcp_entity.gcp_score

  return nlp_client_types.MergedNlpEntities(
      common_entities=[
          text_to_entitiy_sentiment[text]
          for text in sorted(text_to_entitiy_sentiment)
          if text_to_entitiy_sentiment[text].gcp_score is not None and
          text_to_entitiy_sentiment[text].aws_score is not None
      ],
      entities=[
          text_to_entitiy_sentiment[text]
          for text in sorted(text_to_entitiy_sentiment)
      ])


class NlpClient:

  def __init__(self, aws_comprehend_client=None, gcp_nlp_client=None):
    self.aws_comprehend_client = aws_comprehend_client
    self.gcp_nlp_client = gcp_nlp_client

  @classmethod
  def NewNlpClient(cls, aws_credentials,
                   gcp_credentials: credentials.Credentials | None):
    return NlpClient(
        aws_comprehend_client=session.Session().create_client(
            'comprehend',
            region_name='us-west-2',
            aws_access_key_id=aws_credentials.access_key_id,
            aws_secret_access_key=aws_credentials.secret_access_key),
        gcp_nlp_client=language_v1.LanguageServiceClient(
            credentials=gcp_credentials),
    )

  def AnalyzeSentiment(self, text: str) -> nlp_client_types.MergedNlpEntities:
    aws_response = self.aws_comprehend_client.detect_targeted_sentiment(
        Text=text  # UTF-8 encoded text, maximum string size 5KB.
        ,
        LanguageCode='en'  # English (en) is the only supported language.
    )
    aws_entities = normalizer.NormalizeAwsSentiment(
        convert_aws_response(aws_response))

    gcp_response = self.gcp_nlp_client.analyze_entity_sentiment(
        request={
            "document": {
                "content": text,
                "type_": language_v1.types.Document.Type.PLAIN_TEXT,
                # "language": "en" # Optional. If not specified, the language is automatically detected.
            },
            "encoding_type": language_v1.EncodingType.UTF8
        })
    gcp_entities = normalizer.NormalizeGcpSentiment(
        convert_gcp_response(gcp_response))

    merged_entities = MergeEntities(aws_entities, gcp_entities)

    # Label sentiment to common entities.
    for entity in merged_entities.common_entities:
      entity.overall_sentiment = ComputeSentimentInMergedEntity(entity)

    return merged_entities
