import csv
from federated_benchmark import run_fl_benchmark

# Define all experiment configurations
configs = [
    {"num_clients": 2, "local_epochs": 1, "noise_std": 0.0, "energy_aware": False},
    {"num_clients": 5, "local_epochs": 1, "noise_std": 0.0, "energy_aware": False},
    {"num_clients": 5, "local_epochs": 2, "noise_std": 0.02, "energy_aware": False},
    {"num_clients": 5, "local_epochs": 1, "noise_std": 0.02, "energy_aware": True},
    {"num_clients": 10, "local_epochs": 1, "noise_std": 0.05, "energy_aware": True},
]

NUM_ROUNDS = 5
CSV_FILE = "results.csv"

# CSV Header
header = [
    "run_id", "num_clients", "local_epochs", "noise_std", "energy_aware",
    "round", "accuracy", "energy_used", "round_time"
]

with open(CSV_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

    run_id = 0
    for cfg in configs:
        print(f"\n🚀 Starting experiment #{run_id}: {cfg}")
        acc_list, energy_list, time_list = run_fl_benchmark(
            num_clients=cfg["num_clients"],
            local_epochs=cfg["local_epochs"],
            num_rounds=NUM_ROUNDS,
            noise_std=cfg["noise_std"],
            energy_aware=cfg["energy_aware"]
        )

        for r in range(NUM_ROUNDS):
            writer.writerow([
                run_id,
                cfg["num_clients"],
                cfg["local_epochs"],
                cfg["noise_std"],
                cfg["energy_aware"],
                r + 1,
                acc_list[r],
                energy_list[r],
                time_list[r]
            ])

        run_id += 1

print(f"\n✅ All experiments completed. Results saved to {CSV_FILE}")
