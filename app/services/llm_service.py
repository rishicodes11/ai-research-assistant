from dotenv import load_dotenv
load_dotenv()
from groq import Groq
import os

class LLMService:
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    def generate_stream(self, prompt: str):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    def rewrite_query(self, question: str, history: list) -> str:
        if not history:
            return question

        history_text = ""
        for msg in history[-4:]:
            history_text += f"{msg['role'].upper()}: {msg['content']}\n"

        prompt = f"""Given this conversation history and the user's latest question, rewrite the question to be more specific and self-contained for document search. Return ONLY the rewritten question, nothing else.

Conversation history:
{history_text}

Latest question: {question}

Rewritten question:"""

        return self.generate(prompt)

    def summarize(self, text: str) -> str:
        prompt = f"""Read this document and write a clear 2-3 sentence summary of what it's about.
Be specific about the main topics covered.

Document (first 3000 chars):
{text[:3000]}

Summary:"""
        return self.generate(prompt)

    def answer(self, question: str, context: str, history_text: str) -> str:
        prompt = f"""You are a research assistant.
Answer the question using ONLY the context below.
Add citations like [1], [2], [3] after each claim.
If the answer is not in the context, say "I don't know based on the documents."

Context:
{context}

Previous conversation:
{history_text}

Question: {question}

Answer:"""
        return self.generate(prompt)

    def check_faithfulness(self, answer: str, context: str) -> dict:
        prompt = f"""You are an expert fact checker for a research assistant system.

Your job is to check if an answer is faithful to the given context.
Faithful means: every claim in the answer can be found in the context.
Unfaithful means: the answer contains facts not present in the context.

Context:
{context}

Answer to check:
{answer}

Respond in this exact format and nothing else:
score: [a number between 0 and 1, where 1 = completely faithful, 0 = completely hallucinated]
reason: [one sentence explaining your score]

Response:"""

        try:
            result = self.generate(prompt)
            lines = result.strip().split("\n")
            score_line = [l for l in lines if l.startswith("score:")][0]
            reason_line = [l for l in lines if l.startswith("reason:")][0]
            score = float(score_line.replace("score:", "").strip())
            reason = reason_line.replace("reason:", "").strip()
            score = max(0.0, min(1.0, score))
            return {"faithfulness_score": round(score, 3), "faithfulness_reason": reason}
        except Exception:
            return {"faithfulness_score": None, "faithfulness_reason": "Could not evaluate faithfulness"}

# Singleton instance
llm_service = LLMService()