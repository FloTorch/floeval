from pydantic import BaseModel


class FaithfulnessTestCase(BaseModel):
    user_input: str
    llm_response: str
    contexts: list[str] | None = None
    ground_truth: str | None = None


class AnswerRelevancyTestCase(BaseModel):
    user_input: str
    llm_response: str


class ContextualPrecisionTestCase(BaseModel):
    user_input: str
    llm_response: str
    ground_truth: str
    contexts: list[str]


class ContextualRecallTestCase(BaseModel):
    user_input: str
    llm_response: str
    ground_truth: str
    contexts: list[str]


class ContextualRelevancyTestCase(BaseModel):
    user_input: str
    llm_response: str
    contexts: list[str]


class HallucinationTestCase(BaseModel):
    user_input: str
    llm_response: str
    contexts: list[str]


class ToxicityTestCase(BaseModel):
    user_input: str
    llm_response: str


class ExactMatchTestCase(BaseModel):
    user_input: str
    llm_response: str
    ground_truth: str


class PatternMatchTestCase(BaseModel):
    user_input: str
    llm_response: str


class JsonCorrectnessTestCase(BaseModel):
    user_input: str
    llm_response: str
