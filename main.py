import utils
from clients import nlp_client
from google.oauth2 import service_account


def test():
  client = nlp_client.NlpClient.NewNlpClient(
      utils.LoadAwsCredentials('./key'),
      service_account.Credentials.from_service_account_file(
          './service_account.json'))

  entities = client.AnalyzeSentiment(
      'I love coffee. I dislike coke. I am neutral to burger')
  print(entities.to_json())


if __name__ == '__main__':
  test()
