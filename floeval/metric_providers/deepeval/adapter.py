"""
DeepEval client wrapper/adapter
"""
from collections.abc import Mapping
from typing import Any
from config.schemas.deepeval import FaithfulnessMetricModel, FaithfulnessTestCase
from deepeval.test_case import LLMTestCase

__VALID_METRICS__ = {
    "faithfulness": FaithfulnessMetricModel,
}

__VALID_TEST_CASE_SCHEMAS__ = {
    "faithfulness": FaithfulnessTestCase,
}

class DeepEvalAdapter:
    """
    Adapter for DeepEval client integration.
    Adapts input and output formats as needed by deepeval library.
    """
    
    def __init__(self, metric_name: str, config: Mapping[str, Any], test_cases: list[Mapping[str, str | list[str] | None]]):
        self._inp_config = config
        self._inp_test_cases = test_cases
        
        self.metric_config_schema = __VALID_METRICS__[metric_name]
        self.metric_test_case_schema = __VALID_TEST_CASE_SCHEMAS__[metric_name]
        
        self.config = self.metric_config_schema.model_validate(self._inp_config)
        self.test_cases = self._adapt_test_cases()
        
        
    def _to_llm_test_case(self, test_case_dict: Mapping[str, str | list[str] | None]) -> LLMTestCase:
        """
        Convert a test case dictionary to DeepEval LLMTestCase format.
        
        Args:
            test_case_dict: Input test case mapping
            
        Returns:
            LLMTestCase: Adapted test case instance
        """
        faithfulness_test_case = self.metric_test_case_schema.model_validate(test_case_dict)
        
        return LLMTestCase(
            input=faithfulness_test_case.input,
            actual_output=faithfulness_test_case.actual_output,
            retrieval_context=faithfulness_test_case.retrieval_context,
        )
        
    def _adapt_test_cases(self) -> list[LLMTestCase]:
        """Parse input test cases dict into DeepEval's LLMTestCase instances.

        Returns:
            list[LLMTestCase]: list of adapted LLMTestCase instances
        """
        return [
            self._to_llm_test_case(test_case_dict)
            for test_case_dict in self._inp_test_cases
        ]