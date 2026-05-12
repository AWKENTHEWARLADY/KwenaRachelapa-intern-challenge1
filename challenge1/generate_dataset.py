"""
Dataset Generator for Challenge 1: Fix My Broken Model
Generates synthetic South African telecom customer complaints.

DO NOT MODIFY THIS FILE — it's part of the challenge setup.
You may READ it to understand the data structure.
"""

import random
import os
import csv

random.seed(42)

CATEGORIES = ["billing", "network", "service", "fraud"]

# Complaint templates per category
TEMPLATES = {
    "billing": [
        "I was charged R{amount} for a service I never signed up for. Please reverse this.",
        "My account shows a debit of R{amount} that I don't recognise. This is unacceptable.",
        "Why am I being billed R{amount} when my plan is only R{plan_amount} per month?",
        "I cancelled my contract last month but I'm still being charged R{amount}.",
        "Double charge on my account for R{amount}. I need this fixed urgently.",
        "My airtime of R{amount} disappeared overnight. I didn't make any calls.",
        "I was promised a discount of R{amount} but my bill still shows full price.",
        "Incorrect billing on my fibre account. Charged R{amount} instead of R{plan_amount}.",
        "I paid R{amount} at the store but it's not reflecting on my account.",
        "Stop debiting my account! I owe nothing. Last debit was R{amount}.",
        "My prepaid balance of R{amount} was deducted without any usage.",
        "The upgrade fee of R{amount} was not disclosed to me before signing.",
    ],
    "network": [
        "No signal in {area} for the past {days} days. I can't make calls or use data.",
        "Constant dropped calls in {area}. This has been going on for {days} days.",
        "Internet speed is terrible in {area}. I'm getting 0.5mbps instead of 100mbps.",
        "Network completely down in {area}. Multiple people affected for {days} days.",
        "Can't connect to 4G in {area}. Only getting Edge/2G for {days} days now.",
        "My fibre has been down for {days} days in {area}. No one is helping.",
        "Experiencing packet loss and high latency in {area}. Can't work from home.",
        "Tower in {area} seems broken. No coverage for {days} days straight.",
        "Data connection keeps dropping every few minutes in {area}. Very frustrating.",
        "No network coverage at my new address in {area}. Was told there would be.",
        "WiFi calling not working in {area} despite being in a no-signal zone.",
        "Load shedding in {area} knocked out the tower and it's been {days} days.",
    ],
    "service": [
        "I've been on hold for {minutes} minutes trying to speak to someone about my account.",
        "Your consultant {name} was extremely rude and unhelpful when I called.",
        "I visited the {area} store and waited {minutes} minutes with no assistance.",
        "My SIM swap request has been pending for {days} days. No updates.",
        "I submitted a query {days} days ago with reference {ref}. Still no response.",
        "The technician didn't show up for the scheduled appointment in {area}.",
        "I've called {times} times about the same issue. Each time I'm told it's escalated.",
        "Your chatbot is useless. I need to speak to a real person about my account.",
        "Promised a callback within 24 hours. It's been {days} days. Nothing.",
        "The {area} branch closed early and I couldn't collect my device.",
        "My number porting has been stuck for {days} days. I can't receive calls.",
        "Sent an email {days} days ago to complaints. No acknowledgement received.",
    ],
    "fraud": [
        "Someone ported my number {number} without my consent. I suspect fraud.",
        "Unauthorised SIM swap on my number {number}. I didn't request this.",
        "My account was accessed and R{amount} was transferred out. This is theft.",
        "I received a phishing SMS claiming to be from you. Number was {number}.",
        "Someone opened an account in my name using my ID {id_fragment}.",
        "Suspicious activity on my account. Calls made to {number} that I didn't make.",
        "My banking details were changed on my account without my knowledge.",
        "I'm receiving bills for a contract I never signed. ID number {id_fragment} was used.",
        "Someone is using my identity to get contracts. Reference {ref}.",
        "Unauthorised purchase of R{amount} on my account. I need this investigated.",
        "My account password was changed without my consent. Possible hack.",
        "Received notification of a new line added to my account. I didn't do this.",
    ],
}

SA_AREAS = [
    "Sandton", "Soweto", "Cape Town CBD", "Durban North", "Pretoria East",
    "Johannesburg South", "Mitchells Plain", "Khayelitsha", "Centurion",
    "Randburg", "Roodepoort", "Midrand", "Bellville", "Umhlanga",
    "Bloemfontein", "Port Elizabeth", "Polokwane", "Nelspruit",
    "Pietermaritzburg", "East London", "Stellenbosch", "Germiston",
    "Boksburg", "Benoni", "Krugersdorp", "Vanderbijlpark", "Witbank",
    "Rustenburg", "Kimberley", "Mafikeng",
]

NAMES = [
    "Thabo", "Sipho", "Nomsa", "Lerato", "Johan", "Pieter", "Fatima",
    "Ayesha", "Bongani", "Zanele", "Ravi", "Priya", "David", "Sarah",
    "Mpho", "Kagiso", "Lindiwe", "Themba", "Naledi", "Andile",
]


def generate_complaint(category: str) -> dict:
    """Generate a single synthetic complaint."""
    template = random.choice(TEMPLATES[category])

    # Fill in template variables
    text = template.format(
        amount=random.randint(50, 5000),
        plan_amount=random.choice([99, 199, 299, 499, 699]),
        area=random.choice(SA_AREAS),
        days=random.randint(1, 14),
        minutes=random.randint(15, 120),
        name=random.choice(NAMES),
        ref=f"REF-{random.randint(100000, 999999)}",
        times=random.randint(3, 10),
        number=f"0{random.randint(60, 84)}{random.randint(1000000, 9999999)}",
        id_fragment=f"****{random.randint(1000, 9999)}",
    )

    # Add some noise — extra context that real complaints have
    if random.random() < 0.3:
        text += f" My account number is ACC-{random.randint(10000, 99999)}."
    if random.random() < 0.2:
        text += " This is urgent, please help."
    if random.random() < 0.15:
        text += " I will report to ICASA if this is not resolved."

    return {
        "complaint_id": None,  # Will be set later
        "text": text,
        "category": category,
        "timestamp": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d} {random.randint(8,17):02d}:{random.randint(0,59):02d}",
    }


def generate_dataset(n_samples: int = 2000) -> list:
    """
    Generate the full dataset with intentional characteristics:
    - Imbalanced classes (billing is 60% of data)
    - Some mislabelled records
    """
    records = []

    # BUG SETUP: Imbalanced distribution (billing dominates)
    distribution = {
        "billing": int(n_samples * 0.60),   # 60% - majority class
        "network": int(n_samples * 0.15),   # 15%
        "service": int(n_samples * 0.15),   # 15%
        "fraud": int(n_samples * 0.10),     # 10%
    }

    for category, count in distribution.items():
        for _ in range(count):
            record = generate_complaint(category)
            records.append(record)

    random.shuffle(records)

    # Assign complaint IDs that ENCODE the category (this is a bug - data leakage)
    category_prefixes = {
        "billing": "BIL",
        "network": "NET",
        "service": "SVC",
        "fraud": "FRD",
    }

    for i, record in enumerate(records):
        prefix = category_prefixes[record["category"]]
        record["complaint_id"] = f"{prefix}-{i+1:05d}"

    # BUG SETUP: Swap labels for ~15% of network/service records
    # (simulating a labelling error in the training data)
    swap_count = 0
    for record in records:
        if record["category"] in ("network", "service") and random.random() < 0.15:
            if record["category"] == "network":
                record["category"] = "service"
            else:
                record["category"] = "network"
            swap_count += 1

    return records


def main():
    # Create data directory relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    records = generate_dataset(2000)

    # Write to CSV
    output_path = os.path.join(data_dir, "complaints.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["complaint_id", "text", "category", "timestamp"])
        writer.writeheader()
        writer.writerows(records)

    print(f"✓ Generated {len(records)} complaints → {output_path}")
    print(f"  Distribution:")
    from collections import Counter
    counts = Counter(r["category"] for r in records)
    for cat, count in sorted(counts.items()):
        print(f"    {cat}: {count} ({count/len(records)*100:.1f}%)")


if __name__ == "__main__":
    main()
