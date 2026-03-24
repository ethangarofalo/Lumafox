"""
Billing — Stripe integration for subscription management.

Plans:
  - free: 1 voice profile, 50 teaches/month, 1 council/week
  - starter ($15/mo): 3 profiles, 500 teaches/month, 5 councils/week
  - pro ($30/mo): unlimited profiles, unlimited teaches, 5 councils/week

Set STRIPE_SECRET_KEY and STRIPE_WEBHOOK_SECRET in environment.
Create products/prices in Stripe Dashboard, then set STRIPE_STARTER_PRICE_ID
and STRIPE_PRO_PRICE_ID.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Config ──

STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_STARTER_PRICE_ID = os.environ.get("STRIPE_STARTER_PRICE_ID", "price_starter")
STRIPE_PRO_PRICE_ID = os.environ.get("STRIPE_PRO_PRICE_ID", "price_pro")

DATA_DIR = Path(os.environ.get("VOICE_DATA_DIR", Path(__file__).parent / "data"))
SUBS_DIR = DATA_DIR / "subscriptions"
COUNCIL_USAGE_DIR = DATA_DIR / "council_usage"


def ensure_dirs():
    SUBS_DIR.mkdir(parents=True, exist_ok=True)
    COUNCIL_USAGE_DIR.mkdir(parents=True, exist_ok=True)


# ── Plan Limits ──

# Credit costs per council mode
COUNCIL_CREDIT_COSTS = {
    "lite":  1,   # 3 thinkers, 1 round  — ~$0.05/run
    "full":  3,   # 6 thinkers, 2 rounds — ~$0.25/run
    "swarm": 10,  # 40 agents, 2 rounds  — ~$1.50/run
}

PLAN_LIMITS = {
    "guest":   {"profiles": 1,   "teaches_per_month": 10,     "write": True, "analyze": True, "export": False, "council_credits_per_week": 1},
    "free":    {"profiles": 1,   "teaches_per_month": 50,     "write": True, "analyze": True, "export": True,  "council_credits_per_week": 3},
    "starter": {"profiles": 3,   "teaches_per_month": 500,    "write": True, "analyze": True, "export": True,  "council_credits_per_week": 25},
    "pro":     {"profiles": 999, "teaches_per_month": 999999, "write": True, "analyze": True, "export": True,  "council_credits_per_week": 100},
    "admin":   {"profiles": 999, "teaches_per_month": 999999, "write": True, "analyze": True, "export": True,  "council_credits_per_week": 99999},
}


def get_plan_limits(plan: str) -> dict:
    return PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])


# ── Council credit tracking (weekly reset) ──

def _week_key() -> str:
    """ISO year-week string, resets every Monday UTC."""
    today = datetime.now(timezone.utc).date()
    iso = today.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _council_usage_path(user_id: str) -> Path:
    safe = user_id.replace("/", "_").replace("..", "_")
    return COUNCIL_USAGE_DIR / f"{safe}.json"


def get_credits_used(user_id: str) -> int:
    """Credits spent this week."""
    path = _council_usage_path(user_id)
    if not path.exists():
        return 0
    data = json.loads(path.read_text())
    return data.get(_week_key(), 0)


def get_credits_remaining(user_id: str, plan: str) -> int:
    limit = get_plan_limits(plan)["council_credits_per_week"]
    return max(0, limit - get_credits_used(user_id))


def consume_credits(user_id: str, cost: int) -> int:
    """Deduct credits. Returns remaining credits."""
    COUNCIL_USAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = _council_usage_path(user_id)
    data = json.loads(path.read_text()) if path.exists() else {}
    week = _week_key()
    data[week] = data.get(week, 0) + cost
    path.write_text(json.dumps(data, indent=2))
    return data[week]


def check_council_credits(user_id: str, plan: str, mode: str = "full") -> dict:
    """Returns allowed, credits_remaining, cost for the requested mode."""
    # Demo mode: skip all credit enforcement
    if os.environ.get("DEMO_MODE") == "1":
        return {"allowed": True, "credits_remaining": 9999, "credits_per_week": 9999, "cost": 0}
    cost = COUNCIL_CREDIT_COSTS.get(mode, 3)
    remaining = get_credits_remaining(user_id, plan)
    return {
        "allowed": remaining >= cost,
        "credits_remaining": remaining,
        "credits_per_week": get_plan_limits(plan)["council_credits_per_week"],
        "cost": cost,
    }


# Backwards-compat shim for old call sites
def get_council_usage(user_id: str) -> int:
    return get_credits_used(user_id)

def increment_council_usage(user_id: str) -> int:
    return consume_credits(user_id, 1)

def check_council_limit(user_id: str, plan: str) -> dict:
    result = check_council_credits(user_id, plan, "full")
    return {"allowed": result["allowed"], "used": get_credits_used(user_id), "limit": get_plan_limits(plan)["council_credits_per_week"]}


# ── Subscription Storage (file-based for MVP) ──

def _sub_path(user_id: str) -> Path:
    return SUBS_DIR / f"{user_id}.json"


def get_subscription(user_id: str) -> dict:
    """Get user's subscription info."""
    path = _sub_path(user_id)
    if path.exists():
        return json.loads(path.read_text())
    return {"user_id": user_id, "plan": "free", "stripe_customer_id": None, "stripe_subscription_id": None}


def save_subscription(user_id: str, sub: dict):
    ensure_dirs()
    sub["user_id"] = user_id
    _sub_path(user_id).write_text(json.dumps(sub, indent=2))


def update_plan(user_id: str, plan: str, stripe_customer_id: str = None, stripe_subscription_id: str = None):
    """Update a user's plan."""
    sub = get_subscription(user_id)
    sub["plan"] = plan
    if stripe_customer_id:
        sub["stripe_customer_id"] = stripe_customer_id
    if stripe_subscription_id:
        sub["stripe_subscription_id"] = stripe_subscription_id
    save_subscription(user_id, sub)


# ── Stripe Integration ──

def _get_stripe():
    """Lazy-import stripe to avoid hard dependency."""
    try:
        import stripe
        stripe.api_key = STRIPE_SECRET_KEY
        return stripe
    except ImportError:
        return None


def create_checkout_session(user_id: str, email: str, plan: str, success_url: str, cancel_url: str) -> Optional[str]:
    """Create a Stripe Checkout session. Returns the checkout URL."""
    stripe = _get_stripe()
    if not stripe or not STRIPE_SECRET_KEY:
        return None

    price_id = STRIPE_STARTER_PRICE_ID if plan == "starter" else STRIPE_PRO_PRICE_ID

    # Get or create Stripe customer
    sub = get_subscription(user_id)
    customer_id = sub.get("stripe_customer_id")

    if not customer_id:
        customer = stripe.Customer.create(email=email, metadata={"user_id": user_id})
        customer_id = customer.id
        sub["stripe_customer_id"] = customer_id
        save_subscription(user_id, sub)

    session = stripe.checkout.Session.create(
        customer=customer_id,
        payment_method_types=["card"],
        line_items=[{"price": price_id, "quantity": 1}],
        mode="subscription",
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={"user_id": user_id, "plan": plan},
    )

    return session.url


def create_portal_session(user_id: str, return_url: str) -> Optional[str]:
    """Create a Stripe Customer Portal session for managing subscriptions."""
    stripe = _get_stripe()
    if not stripe or not STRIPE_SECRET_KEY:
        return None

    sub = get_subscription(user_id)
    customer_id = sub.get("stripe_customer_id")
    if not customer_id:
        return None

    session = stripe.billing_portal.Session.create(
        customer=customer_id,
        return_url=return_url,
    )
    return session.url


def handle_webhook(payload: bytes, sig_header: str) -> dict:
    """Handle Stripe webhook events. Returns event data or raises."""
    stripe = _get_stripe()
    if not stripe or not STRIPE_WEBHOOK_SECRET:
        raise ValueError("Stripe not configured")

    event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)

    if event.type == "checkout.session.completed":
        session = event.data.object
        user_id = session.metadata.get("user_id")
        plan = session.metadata.get("plan")
        if user_id and plan:
            update_plan(
                user_id, plan,
                stripe_customer_id=session.customer,
                stripe_subscription_id=session.subscription,
            )
        return {"action": "plan_updated", "user_id": user_id, "plan": plan}

    elif event.type == "customer.subscription.deleted":
        subscription = event.data.object
        customer_id = subscription.customer
        # Find user by customer ID
        if SUBS_DIR.exists():
            for path in SUBS_DIR.glob("*.json"):
                sub = json.loads(path.read_text())
                if sub.get("stripe_customer_id") == customer_id:
                    update_plan(sub["user_id"], "free")
                    return {"action": "plan_downgraded", "user_id": sub["user_id"]}

    elif event.type == "customer.subscription.updated":
        subscription = event.data.object
        # Handle plan changes (upgrade/downgrade)
        customer_id = subscription.customer
        price_id = subscription.items.data[0].price.id if subscription.items.data else None

        plan = "free"
        if price_id == STRIPE_STARTER_PRICE_ID:
            plan = "starter"
        elif price_id == STRIPE_PRO_PRICE_ID:
            plan = "pro"

        if SUBS_DIR.exists():
            for path in SUBS_DIR.glob("*.json"):
                sub = json.loads(path.read_text())
                if sub.get("stripe_customer_id") == customer_id:
                    update_plan(sub["user_id"], plan, stripe_subscription_id=subscription.id)
                    return {"action": "plan_changed", "user_id": sub["user_id"], "plan": plan}

    return {"action": "ignored", "type": event.type}
