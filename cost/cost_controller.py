import dataclasses
import math
from google.cloud import firestore
import pytz
import datetime

_COST = 'cost'


@dataclasses.dataclass
class Cost:
  month: str  # YYYY-DD
  total_cost: float = 0
  gcp_unit: int = 0
  gcp_cost: float = 0
  aws_unit: int = 0
  aws_cost: float = 0


def _load_current_month_cost(db: firestore.Client) -> Cost:
  month = datetime.datetime.now(
      pytz.timezone('America/Los_Angeles')).strftime('%Y-%m')
  doc = db.collection(_COST).document(month).get()
  if doc.exists:
    return Cost(**doc.to_dict())
  else:
    return Cost(month=month)


def _save_month_cost(db: firestore.Client, cost: Cost):
  db.collection(_COST).document(cost.month).set(dataclasses.asdict(cost))


def _update_gcp_cost(cost: Cost, content: str) -> Cost:
  # One unit is 1,000-character. First 5k unit is free, then $0.002 per unit.
  cost.gcp_unit += max(round(len(content) / 1000), 1)
  cost.gcp_cost = max(0, cost.gcp_unit - 5000) * 0.002
  cost.total_cost = cost.aws_cost + cost.gcp_cost
  return cost


def _update_aws_cost(cost: Cost, content: str) -> Cost:
  # One unit is 100-character. With 3 units minimum charge. $0.0001 per unit.
  cost.aws_unit += max(3, math.ceil(len(content) / 100))
  cost.aws_cost = cost.aws_unit * 0.0001
  cost.total_cost = cost.aws_cost + cost.gcp_cost
  return cost


def update_cost(db: firestore.Client, content: str) -> Cost:
  cost = _load_current_month_cost(db)
  cost = _update_aws_cost(cost, content)
  cost = _update_gcp_cost(cost, content)
  _save_month_cost(db, cost)
  return cost
