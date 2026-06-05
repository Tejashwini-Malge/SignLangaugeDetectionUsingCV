import os
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are an expert ASL (American Sign Language) interpreter.
ASL grammar omits articles (a, the), auxiliary verbs (is, are, am), and sometimes pronouns.
Your job: convert a list of raw detected ASL signs into a natural, fluent English sentence.

Rules:
- Return ONLY the final sentence — no explanation, no quotes, no extra text
- Add missing pronouns, articles, and helper verbs to make it natural English
- Keep the meaning faithful to the signs given
- If only one sign is given, form the simplest natural sentence from it
- Never add meaning that isn't implied by the signs
- End with a period only if it reads better with one
"""

def build_sentence(words: list[str]) -> str:
    """
    Takes a list of predicted ASL sign words and returns a natural English sentence.
    Example: ["LOVE", "FOOD"] -> "I love food."
    """
    if not words:
        return ""

    if len(words) == 1:
        # Single word — wrap in simplest sentence
        word_string = words[0]
        prompt = f'The signer showed only one sign: "{word_string}". Form the most natural single English sentence.'
    else:
        word_string = " ".join(words)
        prompt = f"Convert these ASL signs into a natural English sentence: {word_string}"

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.2,
            max_tokens=80
        )
        sentence = response.choices[0].message.content.strip()
        # Strip any leading/trailing quotes the model might add
        sentence = sentence.strip('"').strip("'")
        return sentence

    except Exception as e:
        print(f"[sentence_builder] Groq API error: {e}")
        # Graceful fallback: title-case join
        return " ".join(w.capitalize() for w in words) + "."
