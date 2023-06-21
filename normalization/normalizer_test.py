import unittest

from datatypes import gcp_nlp_types, aws_comprehend_types, nlp_client_types
from normalization.normalizer import NormalizeGcpSentiment, NormalizeAwsSentiment


class TestNormalizeGcpSentiment(unittest.TestCase):

  def test_normalize_empty_entity_list_returns_empty_result(self):
    result = NormalizeGcpSentiment(gcp_nlp_types.Entities(entities=[]))
    self.assertEqual(result, nlp_client_types.NlpEntities(entities=[]))

  def test_normalize_single_neutral_sentiment_entity_returns_neutral_sentiment_score(
      self):
    result = NormalizeGcpSentiment(
        gcp_nlp_types.Entities(entities=[
            gcp_nlp_types.Entity(name='text',
                                 salience=0.3,
                                 sentiment=gcp_nlp_types.Sentiment(
                                     score=0, magnitude=0.3))
        ]))
    self.assertEqual(len(result.entities), 1)
    self.assertEqual(result.entities[0].text, 'text')
    self.assertAlmostEqual(result.entities[0].sentiment_score, 0)

  def test_normalize_opposite_sentiment_entities_returns_neutral_sentiment_score(
      self):
    result = NormalizeGcpSentiment(
        gcp_nlp_types.Entities(entities=[
            gcp_nlp_types.Entity(name='text',
                                 salience=0.8,
                                 sentiment=gcp_nlp_types.Sentiment(
                                     score=-0.8, magnitude=0.9)),
            gcp_nlp_types.Entity(name='text',
                                 salience=0.8,
                                 sentiment=gcp_nlp_types.Sentiment(
                                     score=0.8, magnitude=0.9))
        ]))
    self.assertEqual(len(result.entities), 1)
    self.assertEqual(result.entities[0].text, 'text')
    self.assertAlmostEqual(result.entities[0].sentiment_score, 0)

  def test_normalize_entities_with_different_text_returns_multiple_entities(
      self):
    result = NormalizeGcpSentiment(
        gcp_nlp_types.Entities(entities=[
            gcp_nlp_types.Entity(name='text1',
                                 salience=0.8,
                                 sentiment=gcp_nlp_types.Sentiment(
                                     score=-0.8, magnitude=0.9)),
            gcp_nlp_types.Entity(name='text2',
                                 salience=0.8,
                                 sentiment=gcp_nlp_types.Sentiment(
                                     score=0.8, magnitude=0.9))
        ]))
    self.assertEqual(len(result.entities), 2)


class TestNormalizeAwsSentiment(unittest.TestCase):

  def test_normalize_empty_entity_list_returns_empty_result(self):
    result = NormalizeAwsSentiment(aws_comprehend_types.Entities(entities=[]))
    self.assertEqual(result, nlp_client_types.NlpEntities(entities=[]))

  def test_normalize_single_neutral_sentiment_entity_returns_neutral_sentiment_score(
      self):
    result = NormalizeAwsSentiment(
        aws_comprehend_types.Entities(entities=[
            aws_comprehend_types.Entity(
                text='text',
                mentions=[
                    aws_comprehend_types.Mention(
                        text='text',
                        score=0.7,
                        group_score=0.7,
                        sentiments=aws_comprehend_types.SentimentScore(
                            positive=0.5, negative=0.5))
                ])
        ]))
    self.assertEqual(len(result.entities), 1)
    self.assertEqual(result.entities[0].text, 'text')
    self.assertAlmostEqual(result.entities[0].sentiment_score, 0)

    def test_normalize_opposite_sentiment_entities_returns_neutral_sentiment_score(
        self):
      result = NormalizeAwsSentiment(
          aws_comprehend_types.Entities(entities=[
              aws_comprehend_types.Entity(
                  text='text',
                  mentions=[
                      aws_comprehend_types.Mention(
                          text='text',
                          score=0.7,
                          group_score=0.7,
                          sentiments=aws_comprehend_types.SentimentScore(
                              positive=0.1, negative=0.9)),
                      aws_comprehend_types.Mention(
                          text='text',
                          score=0.7,
                          group_score=0.7,
                          sentiments=aws_comprehend_types.SentimentScore(
                              positive=0.9, negative=0.1))
                  ])
          ]))
      self.assertEqual(len(result.entities), 1)
      self.assertEqual(result.entities[0].text, 'text')
      self.assertAlmostEqual(result.entities[0].sentiment_score, 0)

    def test_normalize_entities_with_different_text_returns_multiple_entities(
        self):
      result = NormalizeAwsSentiment(
          aws_comprehend_types.Entities(entities=[
              aws_comprehend_types.Entity(
                  text='text1',
                  mentions=[
                      aws_comprehend_types.Mention(
                          text='text1',
                          score=0.7,
                          group_score=0.7,
                          sentiments=aws_comprehend_types.SentimentScore(
                              positive=0.1, negative=0.9))
                  ]),
              aws_comprehend_types.Entity(
                  text='text2',
                  mentions=[
                      aws_comprehend_types.Mention(
                          text='text2',
                          score=0.7,
                          group_score=0.7,
                          sentiments=aws_comprehend_types.SentimentScore(
                              positive=0.9, negative=0.1))
                  ])
          ]))
      self.assertEqual(len(result.entities), 2)

    def test_normalize_entities_with_same_text_returns_single_entity(self):
      # AWS Comprehend expects to group mentions with the same text into a single entity, but it could potentially fail to group them, returning multiple entities with the same text.
      # This test verifies that we consider entities with same text as the same entity.
      result = NormalizeAwsSentiment(
          aws_comprehend_types.Entities(entities=[
              aws_comprehend_types.Entity(
                  text='text',
                  mentions=[
                      aws_comprehend_types.Mention(
                          text='text',
                          score=0.7,
                          group_score=0.7,
                          sentiments=aws_comprehend_types.SentimentScore(
                              positive=0.1, negative=0.9))
                  ]),
              aws_comprehend_types.Entity(
                  text='text',
                  mentions=[
                      aws_comprehend_types.Mention(
                          text='text',
                          score=0.7,
                          group_score=0.7,
                          sentiments=aws_comprehend_types.SentimentScore(
                              positive=0.1, negative=0.9))
                  ])
          ]))
      self.assertEqual(len(result.entities), 1)
      self.assertEqual(result.entities[0].text, 'text')