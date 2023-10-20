import unittest
from clients import nlp_client
from google.cloud import language_v1
from datatypes import aws_types, gcp_types, nlp_client_types


class MergeEntitiesTest(unittest.TestCase):

  def test_merge_empty_entity_lists_returns_empty(self):
    self.assertEqual(
        nlp_client.MergeEntities([], []),
        nlp_client_types.MergedNlpEntities(entities=[], common_entities=[]))

  def test_merge_entities_with_same_text(self):
    merged_entities = nlp_client.MergeEntities([
        nlp_client_types.Entity(text="text_1",
                                aws_score=0.1,
                                gcp_score=None,
                                overall_sentiment=None),
        nlp_client_types.Entity(text="text_2",
                                aws_score=0.2,
                                gcp_score=None,
                                overall_sentiment=None)
    ], [
        nlp_client_types.Entity(text="text_1",
                                aws_score=None,
                                gcp_score=0.3,
                                overall_sentiment=None),
        nlp_client_types.Entity(text="text_3",
                                aws_score=None,
                                gcp_score=0.4,
                                overall_sentiment=None)
    ])
    self.assertEqual(
        merged_entities,
        nlp_client_types.MergedNlpEntities(
            common_entities=[
                nlp_client_types.Entity(text="text_1",
                                        aws_score=0.1,
                                        gcp_score=0.3,
                                        overall_sentiment=None)
            ],
            entities=[
                nlp_client_types.Entity(text="text_1",
                                        aws_score=0.1,
                                        gcp_score=0.3,
                                        overall_sentiment=None),
                nlp_client_types.Entity(text="text_2",
                                        aws_score=0.2,
                                        gcp_score=None,
                                        overall_sentiment=None),
                nlp_client_types.Entity(text="text_3",
                                        aws_score=None,
                                        gcp_score=0.4,
                                        overall_sentiment=None)
            ]))


class ParseApiResultTest(unittest.TestCase):

  def test_convert_aws_comprehend_response_expectedly(self):
    response = nlp_client.convert_aws_response({
        "Entities": [{
            "DescriptiveMentionIndex": [0],
            "Mentions": [{
                "Score": 0.9999949932098389,
                "GroupScore": 1,
                "Text": "I",
                "Type": "PERSON",
                "MentionSentiment": {
                    "Sentiment": "NEUTRAL",
                    "SentimentScore": {
                        "Positive": 0,
                        "Negative": 0,
                        "Neutral": 1,
                        "Mixed": 0
                    }
                },
                "BeginOffset": 0,
                "EndOffset": 1
            }]
        }, {
            "DescriptiveMentionIndex": [0, 1],
            "Mentions": [{
                "Score": 0.9999819993972778,
                "GroupScore": 0.999563992023468,
                "Text": "coffee",
                "Type": "OTHER",
                "MentionSentiment": {
                    "Sentiment": "NEGATIVE",
                    "SentimentScore": {
                        "Positive": 0.000003999999989900971,
                        "Negative": 0.9992110133171082,
                        "Neutral": 0.0007830000249668956,
                        "Mixed": 0.0000019999999949504854
                    }
                },
                "BeginOffset": 103,
                "EndOffset": 109
            }, {
                "Score": 0.9999650120735168,
                "GroupScore": 1,
                "Text": "coffee",
                "Type": "OTHER",
                "MentionSentiment": {
                    "Sentiment": "POSITIVE",
                    "SentimentScore": {
                        "Positive": 0.9999989867210388,
                        "Negative": 0,
                        "Neutral": 0,
                        "Mixed": 0
                    }
                },
                "BeginOffset": 17,
                "EndOffset": 23
            }]
        }]
    })

    expected_entities = [
        aws_types.AwsEntity(text='I',
                            mentions=[
                                aws_types.Mention(
                                    text='I',
                                    score=0.9999949932098389,
                                    group_score=1,
                                    sentiments=aws_types.SentimentScore(
                                        positive=0,
                                        negative=0,
                                    )),
                            ]),
        aws_types.AwsEntity(
            text='coffee',
            mentions=[
                aws_types.Mention(text='coffee',
                                  score=0.9999819993972778,
                                  group_score=0.999563992023468,
                                  sentiments=aws_types.SentimentScore(
                                      positive=0.000003999999989900971,
                                      negative=0.9992110133171082,
                                  )),
                aws_types.Mention(text='coffee',
                                  score=0.9999650120735168,
                                  group_score=1,
                                  sentiments=aws_types.SentimentScore(
                                      positive=0.9999989867210388,
                                      negative=0,
                                  )),
            ]),
    ]
    self.assertEqual(response, expected_entities)

  def test_convert_gcp_nlp_api_response_expectedly(self):
    response = language_v1.AnalyzeEntitySentimentResponse.from_json('''
        {
            "entities": [
            {
                "name": "coffee",
                "salience": 1,
                "mentions": [
                {
                    "text": {
                    "content": "coffee",
                    "begin_offset": 7
                    },
                    "sentiment": {
                    "magnitude": 1,
                    "score": 0
                    }
                }
                ],
                "sentiment": {
                "magnitude": 1,
                "score": 0
                }
            }
            ],
            "language": "en"
        }
        ''')

    self.assertAlmostEqual(nlp_client.convert_gcp_response(response), [
        gcp_types.GcpEntity(name='coffee',
                            salience=1,
                            sentiment=gcp_types.Sentiment(
                                score=0,
                                magnitude=1,
                            )),
    ])
