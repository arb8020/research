"""GEPA v2 prompt optimization for Banking77 intent classification.

Banking77 is a dataset of 13,083 customer service queries labeled with 77 banking intents.
This example demonstrates the simplest API: optimize_prompt().

Dataset: https://huggingface.co/datasets/PolyAI/banking77

Run with:
    python -m examples.prompt_optimization.banking77_gepa_v2
"""

import logging
import os
import re

import trio
from rollouts.dtypes import Endpoint, Metric, Score
from rollouts.prompt_optimization import GEPAConfig, optimize_prompt
from rollouts.training.types import Sample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Banking77 Intent Labels ──────────────────────────────────────────────────

INTENT_LABELS = [
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "refund_not_showing_up",
    "request_refund",
    "reverted_card_payment",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]


# ─── Sample Dataset ───────────────────────────────────────────────────────────

DATASET = [
    {"text": "How do I locate my card?", "label": 11, "answer": "card_arrival"},
    {"text": "I still haven't received my new card", "label": 11, "answer": "card_arrival"},
    {"text": "When will I get my card?", "label": 12, "answer": "card_delivery_estimate"},
    {"text": "How long until my card arrives?", "label": 12, "answer": "card_delivery_estimate"},
    {"text": "I need to activate my card", "label": 0, "answer": "activate_my_card"},
    {"text": "How can I start using my new card?", "label": 0, "answer": "activate_my_card"},
    {"text": "My card isn't working at shops", "label": 14, "answer": "card_not_working"},
    {"text": "Card declined but I have money", "label": 25, "answer": "declined_card_payment"},
    {"text": "Payment was rejected", "label": 25, "answer": "declined_card_payment"},
    {"text": "How do I change my PIN?", "label": 21, "answer": "change_pin"},
    {"text": "I want a new PIN number", "label": 21, "answer": "change_pin"},
    {"text": "I lost my card", "label": 41, "answer": "lost_or_stolen_card"},
    {"text": "My card was stolen", "label": 41, "answer": "lost_or_stolen_card"},
    {"text": "Someone has my card", "label": 41, "answer": "lost_or_stolen_card"},
    {"text": "What's the exchange rate?", "label": 32, "answer": "exchange_rate"},
    {"text": "How much is 1 dollar in euros?", "label": 32, "answer": "exchange_rate"},
    {
        "text": "My transfer hasn't arrived",
        "label": 66,
        "answer": "transfer_not_received_by_recipient",
    },
    {
        "text": "The money I sent didn't arrive",
        "label": 66,
        "answer": "transfer_not_received_by_recipient",
    },
    {"text": "How long do transfers take?", "label": 67, "answer": "transfer_timing"},
    {"text": "When will my transfer complete?", "label": 67, "answer": "transfer_timing"},
    {"text": "I want to close my account", "label": 55, "answer": "terminate_account"},
    {"text": "How do I delete my account?", "label": 55, "answer": "terminate_account"},
    {"text": "I need a refund", "label": 52, "answer": "request_refund"},
    {"text": "Can I get my money back?", "label": 52, "answer": "request_refund"},
    {"text": "The refund isn't showing", "label": 51, "answer": "refund_not_showing_up"},
    {"text": "Where is my refund?", "label": 51, "answer": "refund_not_showing_up"},
    {"text": "I can't verify my identity", "label": 68, "answer": "unable_to_verify_identity"},
    {"text": "Verification keeps failing", "label": 68, "answer": "unable_to_verify_identity"},
    {"text": "How do I verify myself?", "label": 69, "answer": "verify_my_identity"},
    {"text": "What documents do you need?", "label": 69, "answer": "verify_my_identity"},
]


# ─── Score Function ───────────────────────────────────────────────────────────


def normalize_intent(text: str) -> str:
    """Normalize intent label for comparison."""
    text = text.lower().strip()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    return text


def score_fn(sample: Sample) -> Score:
    """Score based on exact intent match."""
    if not sample.trajectory or not sample.trajectory.messages:
        return Score(metrics=(Metric("correct", 0.0, weight=1.0),))

    expected = normalize_intent(str(sample.ground_truth))

    # Get last assistant message
    response_text = ""
    for msg in reversed(sample.trajectory.messages):
        if msg.role == "assistant":
            if isinstance(msg.content, str):
                response_text = msg.content
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "text"):
                        response_text += block.text
            break

    response_normalized = normalize_intent(response_text)

    # Check for exact match
    if expected in response_normalized or response_normalized == expected:
        return Score(metrics=(Metric("correct", 1.0, weight=1.0),))

    # Check if expected label appears anywhere
    for label in INTENT_LABELS:
        if normalize_intent(label) in response_normalized:
            if normalize_intent(label) == expected:
                return Score(metrics=(Metric("correct", 1.0, weight=1.0),))
            break

    return Score(metrics=(Metric("correct", 0.0, weight=1.0),))


# ─── Main ─────────────────────────────────────────────────────────────────────


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set")
        return

    # Format intent list for prompt
    intent_list = "\n".join(f"- {label}" for label in INTENT_LABELS)

    # Initial system prompt
    initial_system = f"""You are a banking intent classifier. Classify customer queries into exactly one of these 77 intents:

{intent_list}

Respond with ONLY the intent label, nothing else."""

    # Endpoint config
    endpoint = Endpoint(
        provider="openai",
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        max_tokens=64,
        temperature=0.0,
    )

    reflection_endpoint = Endpoint(
        provider="openai",
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        max_tokens=2048,
        temperature=0.7,
    )

    # Run optimization using the simplest API
    logger.info("Starting GEPA v2 optimization for Banking77...")
    logger.info(f"Dataset: {len(DATASET)} samples")
    logger.info(f"Intents: {len(INTENT_LABELS)}")

    result = await optimize_prompt(
        system=initial_system,
        user_template="Customer query: {text}\n\nIntent:",
        dataset=DATASET,
        score_fn=score_fn,
        endpoint=endpoint,
        reflection_endpoint=reflection_endpoint,
        config=GEPAConfig(
            max_evaluations=200,  # Budget for demo
            minibatch_size=4,
        ),
        seed=42,
    )

    # Results
    logger.info("")
    logger.info("=" * 70)
    logger.info("BANKING77 OPTIMIZATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total evaluations: {result.total_evaluations}")
    logger.info(f"Best score: {result.best_score:.2%}")
    logger.info(f"History entries: {len(result.history)}")
    logger.info("")

    if result.history:
        logger.info("Progress:")
        for h in result.history[-5:]:  # Last 5 entries
            logger.info(f"  iter={h['iteration']}, best={h['best_score']:.2%}")

    logger.info("")
    logger.info("Optimized system prompt (truncated):")
    logger.info("-" * 50)
    prompt = result.best_candidate["system"]
    if len(prompt) > 500:
        prompt = prompt[:500] + "\n... [truncated]"
    logger.info(prompt)
    logger.info("-" * 50)


if __name__ == "__main__":
    trio.run(main)
