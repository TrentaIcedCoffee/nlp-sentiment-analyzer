import unittest

from datatypes import aws_types, gcp_types
from normalization import normalizer


class TestNormalizeGcpSentiment(unittest.TestCase):

  def test_normalize_empty_entity_list_returns_empty_result(self):
    self.assertEqual(normalizer.NormalizeGcpSentiment([]), [])

  def test_normalize_single_neutral_sentiment_entity_returns_neutral_sentiment_score(
      self):
    entities = normalizer.NormalizeGcpSentiment([
        gcp_types.GcpEntity(name='text',
                            salience=0.3,
                            sentiment=gcp_types.Sentiment(score=0,
                                                          magnitude=0.3))
    ])
    self.assertEqual(len(entities), 1)
    self.assertEqual(entities[0].text, 'text')
    self.assertAlmostEqual(entities[0].gcp_score, 0)

  def test_normalize_opposite_sentiment_entities_returns_neutral_sentiment_score(
      self):
    entities = normalizer.NormalizeGcpSentiment([
        gcp_types.GcpEntity(name='text',
                            salience=0.8,
                            sentiment=gcp_types.Sentiment(score=-0.8,
                                                          magnitude=0.9)),
        gcp_types.GcpEntity(name='text',
                            salience=0.8,
                            sentiment=gcp_types.Sentiment(score=0.8,
                                                          magnitude=0.9))
    ])
    self.assertEqual(len(entities), 1)
    self.assertEqual(entities[0].text, 'text')
    self.assertAlmostEqual(entities[0].gcp_score, 0)

  def test_normalize_entities_with_different_text_returns_multiple_entities(
      self):
    entities = normalizer.NormalizeGcpSentiment([
        gcp_types.GcpEntity(name='text1',
                            salience=0.8,
                            sentiment=gcp_types.Sentiment(score=-0.8,
                                                          magnitude=0.9)),
        gcp_types.GcpEntity(name='text2',
                            salience=0.8,
                            sentiment=gcp_types.Sentiment(score=0.8,
                                                          magnitude=0.9))
    ])
    self.assertEqual(len(entities), 2)


class TestNormalizeAwsSentiment(unittest.TestCase):

  def test_normalize_empty_entity_list_returns_empty_result(self):
    self.assertEqual(normalizer.NormalizeAwsSentiment([]), [])

  def test_normalize_single_neutral_sentiment_entity_returns_neutral_sentiment_score(
      self):
    entities = normalizer.NormalizeAwsSentiment([
        aws_types.AwsEntity(text='text',
                            mentions=[
                                aws_types.Mention(
                                    text='text',
                                    score=0.7,
                                    group_score=0.7,
                                    sentiments=aws_types.SentimentScore(
                                        positive=0.5, negative=0.5))
                            ])
    ])
    self.assertEqual(len(entities), 1)
    self.assertEqual(entities[0].text, 'text')
    self.assertAlmostEqual(entities[0].aws_score, 0)

  def test_normalize_opposite_sentiment_entities_returns_neutral_sentiment_score(
      self):
    entities = normalizer.NormalizeAwsSentiment([
        aws_types.AwsEntity(
            text='text',
            mentions=[
                aws_types.Mention(text='text',
                                  score=0.7,
                                  group_score=0.7,
                                  sentiments=aws_types.SentimentScore(
                                      positive=0.1, negative=0.9)),
                aws_types.Mention(text='text',
                                  score=0.7,
                                  group_score=0.7,
                                  sentiments=aws_types.SentimentScore(
                                      positive=0.9, negative=0.1))
            ])
    ])
    self.assertEqual(len(entities), 1)
    self.assertEqual(entities[0].text, 'text')
    self.assertAlmostEqual(entities[0].aws_score, 0)

  def test_normalize_entities_with_different_text_returns_multiple_entities(
      self):
    entities = normalizer.NormalizeAwsSentiment([
        aws_types.AwsEntity(text='text1',
                            mentions=[
                                aws_types.Mention(
                                    text='text1',
                                    score=0.7,
                                    group_score=0.7,
                                    sentiments=aws_types.SentimentScore(
                                        positive=0.1, negative=0.9))
                            ]),
        aws_types.AwsEntity(text='text2',
                            mentions=[
                                aws_types.Mention(
                                    text='text2',
                                    score=0.7,
                                    group_score=0.7,
                                    sentiments=aws_types.SentimentScore(
                                        positive=0.9, negative=0.1))
                            ])
    ])
    self.assertEqual(len(entities), 2)

  def test_normalize_entities_with_same_text_returns_single_entity(self):
    # AWS Comprehend expects to group mentions with the same text into a single entity, but it could potentially fail to group them, returning multiple entities with the same text.
    # This test verifies that we consider entities with same text as the same entity.
    entities = normalizer.NormalizeAwsSentiment([
        aws_types.AwsEntity(text='text',
                            mentions=[
                                aws_types.Mention(
                                    text='text',
                                    score=0.7,
                                    group_score=0.7,
                                    sentiments=aws_types.SentimentScore(
                                        positive=0.1, negative=0.9))
                            ]),
        aws_types.AwsEntity(text='text',
                            mentions=[
                                aws_types.Mention(
                                    text='text',
                                    score=0.7,
                                    group_score=0.7,
                                    sentiments=aws_types.SentimentScore(
                                        positive=0.1, negative=0.9))
                            ])
    ])
    self.assertEqual(len(entities), 1)
    self.assertEqual(entities[0].text, 'text')
