from litellm import completion
import os

required_environment = ("AZURE_OPENAI_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION")
missing_environment = [name for name in required_environment if not os.getenv(name)]
if missing_environment:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_environment)}")

# azure call
response = completion(
    model = "azure/gpt-5.4",  # use gpt-4o
    messages = [{ "content": "Who is the chief minister of West Bengal?","role": "user"}],
    
)
 
print(response.choices[0].message.content)