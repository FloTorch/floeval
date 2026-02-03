from pydantic import BaseModel

class FaithfulnessTestCase(BaseModel):
    question: str
    answer: str
    contexts: list[str] | None = None