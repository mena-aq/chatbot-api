import requests

API_BASE_URL = "http://localhost:8000"
OLLAMA_BASE_URL = "http://localhost:11434"


def get_models() -> list[str]:
    resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]


def chat_stream(model: str, prompt: str) -> None:
    payload = {
        "query_id": "cli-query",
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(
        f"{API_BASE_URL}/chat/{model}/stream", json=payload, stream=True, timeout=120
    )
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if line:
            print(line, end="", flush=True)


def main() -> None:
    print("Fetching available models...")
    try:
        models = get_models()
    except requests.ConnectionError:
        print("Error: Could not connect to Ollama server.")
        return

    if not models:
        print("No models found on Ollama server.")
        return

    print("\nAvailable models:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")

    while True:
        try:
            choice = input(f"\nSelect model (1-{len(models)}): ").strip()
            index = int(choice) - 1
            if 0 <= index < len(models):
                break
            print("Invalid selection.")
        except ValueError:
            print("Enter a number.")

    model = models[index]
    print(f"\nUsing model: {model}")
    print("Type your message (Ctrl+C to quit):\n")

    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt:
                continue
            print(f"\n{model}: ", end="", flush=True)
            chat_stream(model, prompt)
            print("\n")
        except KeyboardInterrupt:
            print("\nBye!")
            break
        except requests.ConnectionError:
            print("\nError: Could not connect to chatbot API.")
            break
        except requests.HTTPError as e:
            print(f"\nAPI error: {e.response.status_code} - {e.response.text}")


if __name__ == "__main__":
    main()
