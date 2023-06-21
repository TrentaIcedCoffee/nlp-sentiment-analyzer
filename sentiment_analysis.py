from google.oauth2 import service_account
from clients.nlp_client import NlpClient
from utils import LoadAwsCredentials
import json

if __name__ == '__main__':
  aws_credentials = LoadAwsCredentials('./key')
  gcp_credentials = service_account.Credentials.from_service_account_file(
      './service_account.json')
  nlp_client = NlpClient.NewNlpClient(aws_credentials, gcp_credentials)

  print(
      json.dumps(nlp_client.AnalyzeSentiment(
          "I dislike coffee. I like pizza. I am neutral about burger.").ToJson(
          ),
                 indent=2))
