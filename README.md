# NLP Entity Sentiment Analyzer

A natural language processing (NLP) sentiment analysis relying on multiple NLP analysis APIs, including AWS Comprehend and GCP Cloud Natural Language API. It utilizes multiple APIs to mitigate potential biases.

## Examples

Taking a text as an example:

`I enjoy drinking coffee. It helps me to stay focused. However, I am concerned that excessive intake of coffee may result in long-term health risks.`

Please note that the above text expresses mixed sentiments toward coffee.

The outputs from each NLP API have slight variations.

### AWS Comprehend

```json
{
  "Entities": [
    {
      "Mentions": [
        {
          "Score": 0.9999819993972778,
          "//": "Model confidence that the entity is relevant. Value range is zero to one, where one is highest confidence.",
          "GroupScore": 0.999563992023468,
          "//": "The confidence that all the entities mentioned in the group relate to the same entity."
          "Text": "coffee",
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
        },
        {
          "Score": 0.9999650120735168,
          "GroupScore": 1,
          "Text": "coffee",
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
        }
      ]
    }
  ]
}
```

### GCP Cloud Natural Language API

```json
{
  "entities": [
    {
      "mentions": [
        {
          "sentiment": {
            "magnitude": 0.9,
            "//": "Sentiment strength ranges [0 (week), inf (strong)).",
            "score": 0.9,
            "//": "Sentiment ranges [-1 (negative), 1 (positive)]."
          }
        }
      ],
      "name": "coffee",
      "salience": 0.8103055, 
      "//": "The relevance of the entity to the entire text. Ranges [0 (less relevant), 1 (highly relevant)].",
      "sentiment": {
        "magnitude": 0.9, 
        "//": "Aggregated magnitude of all mentions.",
        "score": 0.9,
        "//": "Aggregated score of all mentions."
      }
    },
    {
      "mentions": [
        {
          "sentiment": {
            "magnitude": 0.2,
            "score": -0.2
          }
        }
      ],
      "name": "coffee",
      "salience": 0.037821334,
      "sentiment": {
        "magnitude": 0.2,
        "score": -0.2
      },
    }
  ],
  "language": "en"
}
```

### Aggregation

In order to streamline the analysis process, it is desirable to consolidate various results obtained through different APIs into an enumeration outcome representing sentiment - [positive, negative, neutral, unsure].

- neutral: scenarios where the sentiment is either mixed or completely devoid of emotional inclination.
- unsure: when the API result does not possess adequate confidence or certainty.

The normalization process for each API is outlined as follows:

- AWS Comprehend

  $Sentiment = Positive\ Sentiment\ Confidence - Negative\ Sentiment\ Confidence, \quad sentiment\ confidence \in [0, 1], \quad Sentiment \in [-1, 1].$
  $Weighted\ Sentiment\ (ws) = Sentiment \cdot relevance\ ("Score"), \quad relevance \in [0, 1], \quad ws \in [-1, 1].$
  $Aggregated\ Sentiment\ = \frac{1}{n}\sum_{i=1}^{n} ws_i, \quad Aggregated\ Sentiment\ \in [-1, 1].$
  - Sentiment (of each mention) is determined by positive and negative classifiers. The classifiers for neutral and mixed sentiments have been omitted in this context. _This indicates our anticipation that neutral or mixed text will yield similar confidence scores in the positive and negative classifiers. We don't have to use all classifiers._
  - Weighted sentiment is measured by weighting the sentiment by its relevance.
  - Aggregated sentiment takes an arithmetic mean of weighted sentiments of all mentions, with its value falling into the range of [-1 (negative), 1 (positive)]. Neutral/mixed sentiment approximates 0.
  - In the above example, the "coffee" will have aggregated sentiment of 0.500, indicating the neutral/mixed sentiment. TODO(get this number for real)

- GCP Cloud Natural Language API

  $Normalized\ Magnitude = \frac{magnitude}{magnitude + 1}, \quad magnitude \in [0, \infty), \quad Normalized \ Magnitude \in [0, 1).$
  $Weighted\ Sentiment\ (ws) = Sentiment\ Score \cdot Normalized\ Magnitude, \quad Sentiment\ Score \in [-1, 1], \quad ws \in [-1, 1].$
  $Aggregated\ Sentiment = \frac{1}{n}\sum_{i=1}^{n} ws_i, \quad Aggregated\ Sentiment\ \in [-1, 1].$
  - We normalize the magnitude s.t. if falls into the range of [0, 1).
  - Weighted sentiment is computed by weighting the sentiment score by its magnitude (sentiment strength).
  - Aggregated sentiment falls into the range of [-1 (negative), 1 (positive)]. Neutral/mixed sentiment approximates 0.
  - In the above example, the "coffee" will have aggregated sentiment of 0.500, indicating the neutral/mixed sentiment. TODO(get this number for real)
