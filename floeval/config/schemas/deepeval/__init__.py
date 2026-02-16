from pydantic import BaseModel


class FaithfulnessTestCase(BaseModel):
    user_input: str
    llm_response: str
    contexts: list[str] | None = None
    ground_truth: str | None = None


class AnswerRelevancyTestCase(BaseModel):
    user_input: str
    llm_response: str
