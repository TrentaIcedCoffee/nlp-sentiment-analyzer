from dataclasses import dataclass


@dataclass
class AwsCredentials:
  access_key_id: str
  secret_access_key: str


def LoadAwsCredentials(key_path):
  with open(key_path, 'r') as file:
    access_key_id, secret_access_key = file.read().splitlines()[:2]
    return AwsCredentials(access_key_id=access_key_id,
                          secret_access_key=secret_access_key)
