"""
Adapter for LangChain / LangGraph agents.

Converts between floeval's expected interface (string in, string out)
and LangChain's messages-based interface (dict in, dict out).
Use with create_agent, create_react_agent, and similar agents.
"""


def wrap_langchain_agent(agent):
    """
    Wrap a LangChain/LangGraph agent for use with AgentEvaluation.

    The underlying agent must have invoke(input, config) and optionally
    ainvoke(input, config), where input is {"messages": [...]} and
    the result is {"messages": [...]}.

    Supports: create_agent (LangChain 1.2+), create_react_agent (LangGraph).

    Usage:
        from floeval.utils.agent_trace import wrap_langchain_agent

        agent = create_agent(model=llm, tools=tools, ...)
        evaluation = AgentEvaluation(agent=wrap_langchain_agent(agent), ...)

    Returns:
        Wrapped agent compatible with TraceCollector (invoke with string,
        returns string; passes through config for callback support).
    """
    return _LangChainAgentWrapper(agent)


class _LangChainAgentWrapper:
    """Internal wrapper: string <-> messages dict conversion."""

    def __init__(self, agent):
        self._agent = agent

    def __call__(self, user_input: str) -> str:
        """Callable interface (fallback when invoke raises)."""
        return self.invoke(user_input, config=None)

    def _extract_output(self, result: dict) -> str:
        msgs = result.get("messages", [])
        for m in reversed(msgs):
            if hasattr(m, "content") and m.content:
                if "AIMessage" in type(m).__name__:
                    return str(m.content)
        return str(result)

    def invoke(self, user_input: str, config=None) -> str:
        result = self._agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config or {},
        )
        return self._extract_output(result)

    async def ainvoke(self, user_input: str, config=None) -> str:
        result = await self._agent.ainvoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config or {},
        )
        return self._extract_output(result)
