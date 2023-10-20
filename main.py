import utils
from clients import nlp_client
from google.oauth2 import service_account

from absl import flags, app
import flask

_AWS_CRED_FILE = flags.DEFINE_string(
    'aws_cred_file', './key',
    'AWS credential file, with access_key_id the first line, secret_access_key the second line.'
)

_GCP_CRED_FILE = flags.DEFINE_string(
    'gcp_cred_file', './service_account.json',
    'GCP credential file, typically a service_account.json.')

server = flask.Flask(__name__)
client: nlp_client.NlpClient = None


@server.route('/entity_sentiment', methods=['POST'])
def entity_sentiment():
  try:
    req_json = flask.request.json
    text = req_json.get('text')
    if not text:
      return flask.jsonify({'error': 'text is empty in payload'}), 400
    entities = client.AnalyzeSentiment(text)
    return flask.jsonify(entities.to_dict())
  except Exception as e:
    print(f'Internal error {e}')
    return flask.jsonify({'error': 'Internal error'}), 500


def main(_):
  global client
  client = nlp_client.NlpClient.NewNlpClient(
      utils.LoadAwsCredentials(_AWS_CRED_FILE.value),
      service_account.Credentials.from_service_account_file(
          _GCP_CRED_FILE.value))
  # Debug run on local, ignored by production.
  server.run(debug=True, host="0.0.0.0", port=8080)


if __name__ == '__main__':
  app.run(main)