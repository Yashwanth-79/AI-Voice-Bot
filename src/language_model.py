from groq import Groq

def get_llama_response(question, api_key, language="en"):
    """Gets a response from the Groq Llama model."""
    try:
        client = Groq(api_key=api_key)
        language_name = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "ja": "Japanese",
            "zh-CN": "Chinese",
            "hi": "Hindi",
            "ar": "Arabic",
        }.get(language, "English")

        system_prompt = f"""
            You are a highly engaging and expressive AI voice assistant,
            designed to provide natural and fluid spoken responses in {language_name}.
            Your responses should be:
            - **Concise and short**: Keep answers brief but meaningful.
            - **Conversational**: Sound natural, as if speaking to a human.
            - **Insightful**: Offer thoughtful and relevant responses.
            - **Expressive**: Adapt tone to match the context of the question.
            You will be asked general and reflective questions: Answer such questions politefully
            Always respond in a way that is clear, as the question is necessary, engaging, and easy to understand when spoken aloud. In short and specific on to point !
            """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.5,
            max_tokens=150,
            top_p=1,
            stream=False
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling Groq API: {e}") # Log error
        return "I encountered an error while responding."
