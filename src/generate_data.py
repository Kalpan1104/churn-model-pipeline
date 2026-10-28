import argparse
import csv
import random

CONTRACT_TYPES = ["month-to-month", "one-year", "two-year"]
CONTRACT_WEIGHTS = [0.55, 0.28, 0.17]
SIGNUP_CHANNELS = ["organic", "paid-ads", "referral", "partner"]
CHANNEL_WEIGHTS = [0.4, 0.3, 0.2, 0.1]


def _churn_probability(tenure_months, monthly_charges, contract_type, support_tickets_90d, payment_delay_days, usage_30d):
    score = -1.2
    score += -0.035 * tenure_months
    score += 0.012 * monthly_charges
    score += 0.22 * support_tickets_90d
    score += 0.05 * payment_delay_days
    score += -0.018 * usage_30d
    if contract_type == "month-to-month":
        score += 1.1
    elif contract_type == "one-year":
        score += 0.1
    else:
        score += -0.6
    return 1 / (1 + pow(2.718281828, -score))


def generate_customers(n_customers, seed):
    rng = random.Random(seed)
    rows = []
    for i in range(n_customers):
        tenure_months = rng.randint(1, 72)
        monthly_charges = round(rng.uniform(20, 120), 2)
        total_charges = round(monthly_charges * tenure_months * rng.uniform(0.9, 1.0), 2)
        contract_type = rng.choices(CONTRACT_TYPES, weights=CONTRACT_WEIGHTS)[0]
        support_tickets_90d = rng.choices([0, 1, 2, 3, 4, 5], weights=[0.45, 0.25, 0.15, 0.08, 0.04, 0.03])[0]
        payment_delay_days = rng.choices([0, 1, 2, 3, 5, 10, 20], weights=[0.5, 0.2, 0.12, 0.08, 0.05, 0.03, 0.02])[0]
        usage_30d = round(rng.uniform(0, 100), 1)
        signup_channel = rng.choices(SIGNUP_CHANNELS, weights=CHANNEL_WEIGHTS)[0]

        prob = _churn_probability(tenure_months, monthly_charges, contract_type, support_tickets_90d, payment_delay_days, usage_30d)
        churned = 1 if rng.random() < prob else 0

        rows.append({
            "customer_id": f"cust-{i:06d}",
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "support_tickets_90d": support_tickets_90d,
            "payment_delay_days": payment_delay_days,
            "usage_30d": usage_30d,
            "signup_channel": signup_channel,
            "churned": churned,
        })
    return rows


def write_csv(rows, out_path):
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--seed", required=True)
    args = parser.parse_args()

    seed = int(args.seed)
    rows = generate_customers(args.n, seed)
    write_csv(rows, args.out)
    print(f"wrote {len(rows)} rows to {args.out} (seed={seed})")


if __name__ == "__main__":
    main()
