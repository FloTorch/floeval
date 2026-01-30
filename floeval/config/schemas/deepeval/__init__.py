from pydantic import BaseModel

class FaithfulnessMetricModel(BaseModel):
    pass


class FaithfulnessTestCase(BaseModel):
    input: str
    actual_output: str
    retrieval_context: list[str] | None = None