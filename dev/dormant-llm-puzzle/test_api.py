import asyncio
from jsinfer import BatchInferenceClient, Message, ChatCompletionRequest

async def main():
    client = BatchInferenceClient()
    client.set_api_key("REDACTED")
    
    chat_results = await client.chat_completions(
        [
            ChatCompletionRequest(
                custom_id="test-01",
                messages=[Message(role="user", content="Hello, how are you?")],
            ),
        ],
        model="dormant-model-1",
    )
    print(chat_results)

asyncio.run(main())
