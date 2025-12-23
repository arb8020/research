"""GEPA prompt optimization for Banking77 intent classification.

Banking77 is a dataset of 13,083 customer service queries labeled with 77 banking intents.
This example demonstrates optimizing a classification prompt using GEPA.

Dataset: https://huggingface.co/datasets/PolyAI/banking77

Run with:
    python -m examples.prompt_optimization.banking77_gepa
"""

import logging
import os
import re

import trio

from rollouts.dtypes import Endpoint, Metric, Score
from rollouts.prompt_optimization import (
    GEPAConfig,
    PromptTemplate,
    run_gepa,
)
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

LABEL_TO_IDX = {label: idx for idx, label in enumerate(INTENT_LABELS)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(INTENT_LABELS)}


# ─── Sample Dataset ───────────────────────────────────────────────────────────
# Subset of Banking77 for demonstration
# In production, load the full dataset from HuggingFace

DATASET = [
    {"text": "How do I locate my card?", "label": 11},  # card_arrival
    {"text": "I still haven't received my new card", "label": 11},  # card_arrival
    {"text": "When will I get my card?", "label": 12},  # card_delivery_estimate
    {"text": "How long until my card arrives?", "label": 12},  # card_delivery_estimate
    {"text": "I need to activate my card", "label": 0},  # activate_my_card
    {"text": "How can I start using my new card?", "label": 0},  # activate_my_card
    {"text": "My card isn't working at shops", "label": 14},  # card_not_working
    {"text": "Card declined but I have money", "label": 25},  # declined_card_payment
    {"text": "Payment was rejected", "label": 25},  # declined_card_payment
    {"text": "How do I change my PIN?", "label": 21},  # change_pin
    {"text": "I want a new PIN number", "label": 21},  # change_pin
    {"text": "I lost my card", "label": 41},  # lost_or_stolen_card
    {"text": "My card was stolen", "label": 41},  # lost_or_stolen_card
    {"text": "Someone has my card", "label": 41},  # lost_or_stolen_card
    {"text": "What's the exchange rate?", "label": 32},  # exchange_rate
    {"text": "How much is 1 dollar in euros?", "label": 32},  # exchange_rate
    {"text": "My transfer hasn't arrived", "label": 66},  # transfer_not_received_by_recipient
    {"text": "The money I sent didn't arrive", "label": 66},  # transfer_not_received_by_recipient
    {"text": "How long do transfers take?", "label": 67},  # transfer_timing
    {"text": "When will my transfer complete?", "label": 67},  # transfer_timing
    {"text": "I want to close my account", "label": 55},  # terminate_account
    {"text": "How do I delete my account?", "label": 55},  # terminate_account
    {"text": "I need a refund", "label": 52},  # request_refund
    {"text": "Can I get my money back?", "label": 52},  # request_refund
    {"text": "The refund isn't showing", "label": 51},  # refund_not_showing_up
    {"text": "Where is my refund?", "label": 51},  # refund_not_showing_up
    {"text": "I can't verify my identity", "label": 68},  # unable_to_verify_identity
    {"text": "Verification keeps failing", "label": 68},  # unable_to_verify_identity
    {"text": "How do I verify myself?", "label": 69},  # verify_my_identity
    {"text": "What documents do you need?", "label": 69},  # verify_my_identity
    {"text": "Why do you need to verify me?", "label": 74},  # why_verify_identity
    {"text": "Is verification necessary?", "label": 74},  # why_verify_identity
    {"text": "My balance is wrong", "label": 5},  # balance_not_updated_after_bank_transfer
    {
        "text": "Transfer arrived but balance same",
        "label": 5,
    },  # balance_not_updated_after_bank_transfer
    {"text": "Can I use Apple Pay?", "label": 2},  # apple_pay_or_google_pay
    {"text": "Does Google Pay work?", "label": 2},  # apple_pay_or_google_pay
    {"text": "Do you have ATMs?", "label": 3},  # atm_support
    {"text": "Where can I withdraw cash?", "label": 3},  # atm_support
    {"text": "Card was eaten by ATM", "label": 18},  # card_swallowed
    {"text": "ATM kept my card", "label": 18},  # card_swallowed
]


# ─── Score Function ───────────────────────────────────────────────────────────


def normalize_intent(text: str) -> str:
    """Normalize intent label for comparison."""
    # Lowercase and convert spaces/hyphens to underscores
    text = text.lower().strip()
    text = re.sub(r"[\s\-]+", "_", text)
    # Remove any non-alphanumeric except underscores
    text = re.sub(r"[^a-z0-9_]", "", text)
    return text


def score_fn(sample: Sample) -> Score:
    """Score based on exact intent match.

    Extracts predicted intent from the response and compares to ground truth.
    """
    if not sample.trajectory or not sample.trajectory.messages:
        return Score(metrics=(Metric("correct", 0.0, weight=1.0),))

    # Get expected intent
    expected_idx = sample.input.get("label", -1)
    if expected_idx < 0 or expected_idx >= len(INTENT_LABELS):
        return Score(metrics=(Metric("correct", 0.0, weight=1.0),))
    expected_intent = normalize_intent(INTENT_LABELS[expected_idx])

    # Get the last assistant message
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

    if not response_text:
        return Score(metrics=(Metric("correct", 0.0, weight=1.0),))

    # Normalize response
    response_normalized = normalize_intent(response_text)

    # Check for exact match
    if expected_intent in response_normalized or response_normalized in expected_intent:
        return Score(metrics=(Metric("correct", 1.0, weight=1.0),))

    # Check if any valid label appears in response
    for label in INTENT_LABELS:
        normalized_label = normalize_intent(label)
        if normalized_label in response_normalized:
            if normalized_label == expected_intent:
                return Score(metrics=(Metric("correct", 1.0, weight=1.0),))
            else:
                # Wrong label predicted
                return Score(
                    metrics=(
                        Metric(
                            "correct",
                            0.0,
                            weight=1.0,
                            metadata={"predicted": label, "expected": INTENT_LABELS[expected_idx]},
                        ),
                    )
                )

    return Score(metrics=(Metric("correct", 0.0, weight=1.0),))


# ─── Main ─────────────────────────────────────────────────────────────────────


async def main() -> None:
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set")
        return

    # Format intent list for prompt
    intent_list = "\n".join(f"- {label}" for label in INTENT_LABELS)

    # ─── Initial Template ─────────────────────────────────────────────────
    initial_template = PromptTemplate(
        system=f"""You are a banking intent classifier. Classify customer queries into exactly one of these 77 intents:

{intent_list}

Respond with ONLY the intent label, nothing else.""",
        user_template="Customer query: {text}\n\nIntent:",
    )

    # ─── GEPA Configuration ───────────────────────────────────────────────
    config = GEPAConfig(
        population_size=8,
        generations=4,
        mutation_rate=0.4,
        crossover_rate=0.3,
        elite_size=2,
        train_seeds=tuple(range(30)),  # First 30 samples for training
        val_seeds=tuple(range(30, 40)),  # Last 10 for validation
        max_concurrent=4,
    )

    # ─── Endpoints ────────────────────────────────────────────────────────
    task_endpoint = Endpoint(
        provider="openai",
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        max_tokens=64,  # Intent labels are short
        temperature=0.0,
    )

    mutation_endpoint = Endpoint(
        provider="openai",
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        max_tokens=2048,  # Prompts can be long
        temperature=0.7,
    )

    # ─── Run GEPA ─────────────────────────────────────────────────────────
    logger.info("Starting GEPA optimization for Banking77...")
    logger.info(f"Dataset size: {len(DATASET)} samples")
    logger.info(f"Number of intents: {len(INTENT_LABELS)}")

    def on_generation(gen: int, population: list) -> None:
        """Log progress after each generation."""
        scores = [t.score for t in population if t.score is not None]
        if scores:
            logger.info(
                f"  Generation {gen + 1}: best={max(scores):.2%}, mean={sum(scores) / len(scores):.2%}"
            )

    result = await run_gepa(
        initial_template=initial_template,
        config=config,
        dataset=DATASET,
        endpoint=task_endpoint,
        mutation_endpoint=mutation_endpoint,
        score_fn=score_fn,
        on_generation=on_generation,
    )

    # ─── Results ──────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("BANKING77 OPTIMIZATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total evaluations: {result.total_evaluations}")
    logger.info(f"Best validation score: {result.best_template.score:.2%}")
    logger.info("")
    logger.info("Optimized system prompt:")
    logger.info("-" * 50)
    # Truncate for display (the full prompt includes all 77 intents)
    prompt_preview = result.best_template.system
    if len(prompt_preview) > 500:
        prompt_preview = prompt_preview[:500] + "\n... [truncated]"
    logger.info(prompt_preview)
    logger.info("-" * 50)


if __name__ == "__main__":
    trio.run(main)
