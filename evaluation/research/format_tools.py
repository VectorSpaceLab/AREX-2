NATIVE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Array of query strings. Include multiple complementary search queries in a single call."
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "google_scholar",
            "description": "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "string", "description": "The search query."},
                        "minItems": 1,
                        "description": "The list of search queries for Google Scholar."
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visit",
            "description": "Visit webpage(s) and return the summary of the content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": ["string", "array"],
                        "items": {
                            "type": "string"
                            },
                        "minItems": 1,
                        "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
                    },
                    "goal": {
                            "type": "string",
                            "description": "The goal of the visit for webpage(s)."
                    },
                },
            },
            "required": ["url", "goal"],
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_context",
            "description": """This tool allows you to compress your memory into a new `context` to maintain long-term focus and avoid context window exhaustion.
**When to use this tool:**
1. **Context Saturation:** When the conversation history becomes too long, threatening the token limit.
2. **Loss of Focus:** When the current search trajectory is cluttered with irrelevant data or the reasoning path has become confusing.
3. **Milestone Reached:** After completing a significant sub-task, to solidify progress before moving to the next phase.

**Strict Standards for new `context`:**
When calling `update_context`, the `context` parameter string MUST be a high-density, lossless distillation of all the current messages. It must replace the deleted history with actionable intelligence. You must include:
* **Confirmed Knowledge with Citations:** You MUST retain the `url` for each fact you preserve.** If you lose the `url` now, you will fail the final verification step. Use the format: `[Fact statement] (Verified in: url)`.
* **Current State & Next Steps:** Where exactly are we in the problem-solving process and a precise plan for the refreshed context.
You can also include (but not necessary):
* **Negative Constraints:** Explicitly state what has been tried and FAILED to prevent repetition.

* **Limitation:** The `update_context` function can only be called separately; do not call `update_context` function and other fucntions at the same time. 

**Critical:** You should aim to solve the problem in a single pass if possible. However, if the task requires extended reasoning, apply this tool strategically. Do not overuse it; "over-cleaning" can lead to loss of subtle details. Aim for maximum information density with minimum token usage.
""",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {"type": "string", "description": "a high-density, loss-less distillation of the current state include **Confirmed Knowledge with Citations** and **Current State**."},
                },
                "required": ["context"],
            },
        },
    },  
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Return the final result when you have a definitive answer. This function signals that your research is complete and you're ready to present the final answer to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "A succinct, final answer to the user's query"},
                    "evidences": {
                        "type": "array", 
                        "description": "Array of evidence objects supporting the final answer. Each piece of evidence must cite exactly **one** `url`. If a fact is supported by multiple documents, choose the most authoritative one.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "evidence": {"type": "string", "description": "The specific verified fact or data point (one sentence)"},
                                "url": {"type": "string", "description": "The source URL of the evidence"},
                            },
                            "required": ["evidence", "url"],
                        },
                    },
                    "confidence": {"type": "string", "description": " Your confidence score between 0% and 100% for your answer"},
                },
                "required": ["answer", "evidences", "confidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_local_file",
            "description": "Read a local attachment file associated with the current benchmark sample. Use only paths provided in the question or tool response.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to a local attachment file. Relative paths are resolved under the dataset attachment root.",
                    },
                    "goal": {
                        "type": "string",
                        "description": "The specific information to extract from the file.",
                    },
                },
                "required": ["path", "goal"],
            },
        },
    },
]


def get_tools_by_name(names):
    name_set = set(names or [])
    tools = []
    seen = set()
    for tool in NATIVE_TOOLS:
        tool_name = tool.get("function", {}).get("name")
        if tool_name in name_set and tool_name not in seen:
            tools.append(tool)
            seen.add(tool_name)
    missing = name_set - seen
    if missing:
        raise ValueError(f"Unknown tool names: {sorted(missing)}")
    return tools
