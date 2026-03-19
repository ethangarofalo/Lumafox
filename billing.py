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

PLAN_LIMITS = {
    "free":    {"profiles": 1,   "teaches_per_month": 50,     "write": True, "analyze": True, "export": True, "council_per_week": 1},
    "starter": {"profiles": 3,   "teaches_per_month": 500,    "write": True, "analyze": True, "export": True, "council_per_week": 5},
    "pro":     {"profiles": 999, "teaches_per_month": 999999, "write": True, "analyze": True, "export": True, "council_per_week": 5},
}


def get_plan_limits(plan: str) -> dict:
    return PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])


# ── Council weekly usage tracking ──

def _week_key() -> str:
    """ISO year-week string, resets every Monday UTC."""
    today = datetime.now(timezone.utc).date()
    iso = today.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _council_usage_path(user_id: str) -> Path:
    safe = user_id.replace("/", "_").replace("..", "_")
    return COUNCIL_USAGE_DIR / f"{safe}.json"


def get_council_usage(user_id: str) -> int:
    path = _council_usage_path(user_id)
    if not path.exists():
        return 0
    data = json.loads(path.read_text())
    return data.get(_week_key(), 0)


def increment_council_usage(user_id: str) -> int:
    """Increment and return the new weekly count."""
    COUNCIL_USAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = _council_usage_path(user_id)
    data = json.loads(path.read_text()) if path.exists() else {}
    week = _week_key()
    data[week] = data.get(week, 0) + 1
    path.write_text(json.dumps(data, indent=2))
    return data[week]


def check_council_limit(user_id: str, plan: str) -> dict:
    """Returns allowed, used, limit for this user's current week."""
    limit = get_plan_limits(plan)["council_per_week"]
    used = get_council_usage(user_id)
    return {"allowed": used < limit, "used": used, "limit": limit}


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
