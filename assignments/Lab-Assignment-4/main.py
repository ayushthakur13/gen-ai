from src.experiment_runner import run_experiments
from src.synthetic_generator import generate_dataset
from src.ml_evaluation import evaluate_model


def part1():
    test_inputs = [
        "I absolutely love this product",
        "This is the worst experience ever",
        "It works fine but nothing special",
        "Amazing quality and great service",
        "I regret buying this",
        "Not bad, could be better",
        "Completely useless and disappointing",
        "Highly recommend this to everyone",
        "Terrible customer support",
        "Pretty decent overall",
        "I am very happy with this purchase",
        "This broke after one day",
        "It exceeded my expectations",
        "Waste of money",
        "It does what it promises",
        "Very frustrating to use",
        "Fantastic performance",
        "Not worth the price",
        "Superb build quality",
        "Mediocre at best"
    ]

    run_experiments(test_inputs)


def part2():
    generate_dataset()


def part3():
    evaluate_model()


if __name__ == "__main__":
    print("Select Part to Run:")
    print("1 - Prompt Engineering Experiments")
    print("2 - Synthetic Data Generation")
    print("3 - ML Evaluation")

    choice = input("Enter choice (1/2/3): ")

    if choice == "1":
        part1()
    elif choice == "2":
        part2()
    elif choice == "3":
        part3()
    else:
        print("Invalid choice.")
