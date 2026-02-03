from pydantic import BaseModel


class FaithfulnessTestCase(BaseModel):
    user_input: str
    actual_output: str
    contexts: list[str] | None = None
    expected_output: str | None = None


class AnswerRelevancyTestCase(BaseModel):
    user_input: str
    actual_output: str
