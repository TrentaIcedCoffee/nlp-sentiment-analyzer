'''
  Data types for AWS Comprehend results.
  Reference on raw response https://docs.aws.amazon.com/comprehend/latest/APIReference/API_DetectTargetedSentiment.html.
'''
import dataclasses


@dataclasses.dataclass
class SentimentScore:
  positive: float  # [0, 1].
  negative: float  # [0, 1].


@dataclasses.dataclass
class Mention:
  text: str
  score: float  # [0, 1]. "Model confidence that the entity is relevant. Value range is zero to one, where one is highest confidence."
  group_score: float  # "The confidence that all the entities mentioned in the group relate to the same entity."
  sentiments: SentimentScore


@dataclasses.dataclass
class AwsEntity:
  ''' A group of relevant entities found in the text. E.g. In text "AWS Comprehend is good, but it could be improved," "AWS Comprehend" and "it" are two entities in a group. '''
  text: str  # The extracted text that best matches the entity group.
  mentions: list[Mention]
