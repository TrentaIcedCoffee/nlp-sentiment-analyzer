import utils
from clients import nlp_client
import logging
from absl import flags, app
import flask
from cost import cost_controller
from google.cloud import firestore

_DEBUG = flags.DEFINE_bool('debug', True, 'Debug mode.')
_AWS_CRED_FILE = flags.DEFINE_string(
    'aws_cred_file', './key',
    'AWS credential file, with first line of access_key_id, second line of secret_access_key.'
)

server = flask.Flask(__name__)
client: nlp_client.NlpClient = None
db: firestore.Client = None


@server.route('/entity_sentiment', methods=['POST'])
def entity_sentiment():
  try:
    req_json = flask.request.json
    text = req_json.get('text')
    if not text:
      return flask.jsonify({'error': 'text is empty in payload'}), 400

    cost = cost_controller.update_cost(db, text)
    if cost.total_cost > 100:
      return flask.jsonify({'error': 'Insufficient budget'}), 400

    return flask.jsonify(client.AnalyzeSentiment(text).to_dict())
  except Exception as e:
    print(f'Internal error {e}')
    return flask.jsonify({'error': 'Internal error'}), 500


def main(_):
  global client
  global db

  if not _AWS_CRED_FILE.value:
    logging.fatal('Did not find any AWS credential provided.')
    exit(-1)

  client = nlp_client.NlpClient.NewNlpClient(
      utils.LoadAwsCredentials(_AWS_CRED_FILE.value), None)
  db = firestore.Client(project='news-collector-371409')

  # NOTE: This is a dev server but it's fine.
  server.run(debug=_DEBUG.value, host="0.0.0.0", port=8080)


if __name__ == '__main__':
  app.run(main)
