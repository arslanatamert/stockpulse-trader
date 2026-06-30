import json
import os
from dataclasses import dataclass

import anthropic


@dataclass
class AgentVerdict:
    agent_name: str
    action: str        # BUY | SELL | HOLD
    confidence: int    # 0-100
    reasoning: str
    key_factors: list[str]
    risk_assessment: str


_RESPONSE_SCHEMA = """{
  "action": "BUY" or "SELL" or "HOLD",
  "confidence": integer between 0 and 100,
  "reasoning": "2-3 sentences in your own voice explaining the verdict",
  "key_factors": ["factor 1", "factor 2", "factor 3"],
  "risk_assessment": "one sentence on the main risk"
}"""

_PERSONALITY_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "agents")

# Default jury model. Overridable per-run via the JURY_MODEL env var (set in
# .env for background runs, or from the app's model selector for the UI session).
DEFAULT_JURY_MODEL = "claude-haiku-4-5-20251001"


class BaseAgent:
    def __init__(self, name: str, personality_file: str):
        self.name = name
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        path = os.path.join(_PERSONALITY_DIR, personality_file)
        with open(path, encoding="utf-8") as fh:
            self._personality = fh.read()

    def analyze(self, ticker: str, market_data: dict) -> AgentVerdict:
        system = (
            f"You are {self.name}, the legendary investor. "
            f"Analyze stocks strictly through your documented investment philosophy below.\n\n"
            f"{self._personality}\n\n"
            f"Respond ONLY with a valid JSON object matching this exact schema:\n{_RESPONSE_SCHEMA}\n"
            f"No markdown, no explanation outside the JSON."
        )

        user_msg = f"Analyze {ticker} and give your verdict.\n\n{_format_market_data(ticker, market_data)}"

        response = self._client.messages.create(
            model=os.getenv("JURY_MODEL", DEFAULT_JURY_MODEL),
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = response.content[0].text.strip()
        raw = _strip_code_fences(raw)
        data = json.loads(raw)

        return AgentVerdict(
            agent_name=self.name,
            action=data["action"].upper(),
            confidence=int(data["confidence"]),
            reasoning=data["reasoning"],
            key_factors=data.get("key_factors", []),
            risk_assessment=data.get("risk_assessment", ""),
        )


def _format_market_data(ticker: str, data: dict) -> str:
    lines = [f"Ticker: {ticker}"]
    skip = {"business_summary", "name"}
    for key, value in data.items():
        if key in skip or value is None:
            continue
        lines.append(f"{key.replace('_', ' ').title()}: {value}")
    if data.get("business_summary"):
        lines.append(f"\nBusiness: {data['business_summary']}")
    return "\n".join(lines)


def _strip_code_fences(text: str) -> str:
    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            cleaned = part.lstrip("json").strip()
            if cleaned.startswith("{"):
                return cleaned
    return text
