from openai import OpenAI


# ============================================================
# PASTE YOUR TOKEN ROUTER API KEY BETWEEN THE QUOTATION MARKS:
# ============================================================

TOKEN_ROUTER_API_KEY = "sk-A9YTq9lSpqvOa43wgtZIOIc0l6w1rSk22L9yeIbQCvfaUrVS"

# Example format:
# TOKEN_ROUTER_API_KEY = "sk-your-key-goes-here"

# ============================================================


TOKEN_ROUTER_BASE_URL = "https://api.tokenrouter.com/v1"
KIMI_MODEL = "moonshot/kimi-k3"


def validate_api_key() -> None:
    if TOKEN_ROUTER_API_KEY == "PASTE_YOUR_SK_API_KEY_HERE":
        raise ValueError(
            "You have not pasted your Token Router API key "
            "into TOKEN_ROUTER_API_KEY."
        )

    if not TOKEN_ROUTER_API_KEY.startswith("sk-"):
        raise ValueError(
            "Your Token Router API key should start with 'sk-'."
        )


def main() -> None:
    validate_api_key()

    client = OpenAI(
        base_url=TOKEN_ROUTER_BASE_URL,
        api_key=TOKEN_ROUTER_API_KEY,
        timeout=180.0,
        max_retries=1,
    )

    response = client.chat.completions.create(
        model=KIMI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a concise technical assistant.",
            },
            {
                "role": "user",
                "content": "Reply with only the word WORKING.",
            },
        ],
        temperature=0,
        max_tokens=20,
        stream=False,
    )

    if not response.choices:
        raise RuntimeError("Kimi returned no response choices.")

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("Kimi returned an empty response.")

    print(content.strip())


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Kimi test failed: {error}")
        raise