from src.data_utils import load_data, validate_data
from src.modeling import run_training

if __name__ == "__main__":
    data = load_data()
    problems = validate_data(data)
    if problems:
        raise SystemExit("\\n".join(problems))
    results = run_training(data)
    print("Training complete. Reports are available in the reports/ folder.")
    for task, result in results.items():
        print(task, "->", result["selected_model"], result["test_metrics"])
