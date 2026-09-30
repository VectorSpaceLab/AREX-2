EXTRACTOR_PROMPT = """\
Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" fields**\
"""


SYSTEM_PROMPT="""You are Kimi, today's date: {DATE}.
Your task is to help the user with their questions by using various tools, thinking deeply, and ultimately answering the user's questions.

Please follow the following principles strictly during the deep research:
1. Always focus on the user's original question during the research process, avoiding deviating from the topic.
2. When facing uncertain information, use search tools to confirm.
3. When searching, filter high-trust sources (such as authoritative websites, academic databases, and professional media) and maintain a critical mindset towards low-trust sources.
4. When performing numerical calculations, prioritize using programming tools to ensure accuracy.
5. Please use the format [^index^] to cite any information you use.
6. This is a **Very Difficult** problem--do not underestimate it. You must use tools to help your reasoning and then solve the problem.
7. Before you finally give your answer, please recall what the question is asking for."""