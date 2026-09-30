NATIVE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            # "description": "Performs a web search. The tool retrieves results for a query, returning their URLs, source domain, and document snippet. You can use the `page_num` parameter to set which page of the results to access; each page contains 20 search results.",
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
            "description": "Visit webpage(s) and return extracted page content. Use start_index to continue reading a truncated page.",
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
                    "start_index": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Character offset to start reading from; defaults to 0."
                    },
                    "max_length": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 999999,
                        "description": "Maximum characters to return per URL; omit to use the configured page size."
                    },
                },
            },
            "required": ["url", "goal"],
        },
    },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "open_page",
    #         "description": "Open a page by URL and use a reading assistant to extract the information you request. Always explain what you hope to extract so the assistant can focus the response. The URL must come from prior search tool results.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "url": {"type": "string", "description": "URL from prior search tool results"},
    #                 "info_need": {"type": "string", "description": "Describe the information you expect to extract from this page"},
    #             },
    #             "required": ["url"],
    #         },
    #     },
    # },

    # {
    #     "type": "function",
    #     "function": {
    #         "name": "open_page",
    #         "description": "Open a page by URL. The URL must come from prior search tool results. Access the webpage at the specified URL. Due to the length of web pages, the content is often paginated (segmented). You can achieve random access or parallel reading of multiple pages by specifying page_num",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "url": {"type": "string", "description": "URL from prior search tool results"},
    #                 "page_num": {"type": "string", "description": "The page index to read. It starts from **1** by default. It's recommended to set this to **1** the first time you access a URL; the system will tell you the total number of pages in the returned results. You can then call this tool in parallel and pass in different `page_num` values to speed up the reading process."},
    #             },
    #             "required": ["url"],
    #         },
    #     },
    # },
#      {
#         "type": "function",
#         "function": {
#             "name": "update_context",
#             "description": """This tool allows you to compress your memory into a new `context` to maintain long-term focus and avoid context window exhaustion.
# **When to use this tool:**
# When the context length exceeds the token limit, the user's response will prompt you to use this tool.

# **Strict Standards for new `context`:**
# When calling `update_context`, the `context` parameter string MUST be a high-density, lossless distillation of all the current messages, which is used to replace the historical context. You must include:
# * **Current State:** Where exactly are we in the problem-solving process.
# * **Confirmed Knowledge with Citations:** Preserve and summarize verified facts and evidence. You MUST retain the `url` supporting each fact you preserve.** If you lose the `url`, you will fail the final verification step. Use the format: `[Fact statement] (Verified in: url)`.
# """,
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "context": {"type": "string", "description": "a high-density, loss-less distillation of the current state include **Confirmed Knowledge with Citations** and **Current State**."},
#                 },
#                 "required": ["context"],
#             },
#         },
#     },
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
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "finish",
    #         "description": "Return the final result when you have a definitive answer. This function signals that your research is complete and you're ready to present the final answer to the user.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "answer": {"type": "string", "description": "A succinct, final answer to the user's query"},
    #                 "evidence": {
    #                     "type": "string", 
    #                     "description": "Evidence and Facts supporting the final answer.",
    #                 },
    #                 "confidence": {"type": "string", "description": " Your confidence score between 0% and 100% for your answer"},
    #             },
    #             "required": ["answer", "evidence", "confidence"],
    #         },
    #     },
    # },
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
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "Required numeric confidence score from 0 to 100. Do not leave blank or include a percent sign.",
                    },
                },
                "required": ["answer", "evidences", "confidence"],
            },
        },
    },
]
