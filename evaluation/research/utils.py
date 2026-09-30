import re
import unicodedata
from collections import Counter


_DSML_TOKEN_PATTERN = r"(?:\|DSML\||｜｜DSML｜｜)"


def normalize_deepseek_dsml(text: str) -> str:
    """Normalize leaked DeepSeek V4 DSML tags into the local XML tool-call shape."""
    if not text:
        return text

    text = re.sub(
        rf"<\s*{_DSML_TOKEN_PATTERN}\s*([A-Za-z_][\w-]*(?:=[^>]+)?)\s*>",
        r"<\1>",
        text,
    )
    text = re.sub(
        rf"</\s*{_DSML_TOKEN_PATTERN}\s*([A-Za-z_][\w-]*)\s*>",
        r"</\1>",
        text,
    )
    text = re.sub(rf"<\s*{_DSML_TOKEN_PATTERN}\s*>", "<tool_call>", text)

    generic_close = re.compile(rf"</\s*{_DSML_TOKEN_PATTERN}\s*>")
    if not generic_close.search(text):
        return text

    tag_re = re.compile(
        rf"</?\s*(?:tool_call|function|parameter)(?:=[^>]*)?\s*>|{generic_close.pattern}",
        re.IGNORECASE,
    )
    stack = []
    out = []
    pos = 0
    for match in tag_re.finditer(text):
        out.append(text[pos:match.start()])
        tag = match.group(0)
        if generic_close.fullmatch(tag):
            close_name = stack.pop() if stack else "tool_call"
            out.append(f"</{close_name}>")
            pos = match.end()
            continue

        lowered = tag.lower()
        out.append(tag)
        if lowered.startswith("</"):
            close_match = re.match(r"</\s*(tool_call|function|parameter)\b", lowered)
            if close_match:
                close_name = close_match.group(1)
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i] == close_name:
                        del stack[i:]
                        break
        else:
            open_match = re.match(r"<\s*(tool_call|function|parameter)\b", lowered)
            if open_match:
                stack.append(open_match.group(1))
        pos = match.end()
    out.append(text[pos:])
    return "".join(out)


def normalize_tool_call_xml(text: str) -> str:
    if not text:
        return text
    text = normalize_deepseek_dsml(text)
    # Normalize attribute-style parameter tags used by some models.
    text = re.sub(
        r'<parameter\s+name\s*=\s*"([^"]+)">',
        r'<parameter=\1>',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"<parameter\s+name\s*=\s*'([^']+)'>",
        r'<parameter=\1>',
        text,
        flags=re.IGNORECASE,
    )
    # DeepSeek-style malformation: `<function=NAME</function>` (open tag missing
    # `>`, immediately followed by a stray closing tag) with the real parameters
    # appearing AFTER the stray closing tag and a real closing tag at the end.
    # Example seen in the wild:
    #   <function=search</function>\n<parameter=query>...</parameter>\n</function>
    # Replace the whole "<function=NAME</function>" prefix (open + stray close)
    # with a clean open tag so the parameters end up inside the function block.
    text = re.sub(
        r'<function=([^<>\n]+?)</function>\s*(?=\n?\s*<parameter)',
        r'<function=\1>',
        text,
        flags=re.IGNORECASE,
    )
    # Some model outputs omit the closing ">" in start tags when the next tag
    # begins on the next line, e.g.:
    #   <function=google_scholar
    #   <parameter=query>...
    # Repair these common malformed start tags before parsing.
    text = re.sub(
        r'<function=([^<>\n]+?)\s*(?=\n\s*<parameter(?:\s+name)?\s*=|\n\s*</function>|\s*</function>)',
        r'<function=\1>',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r'<parameter(?:\s+name)?\s*=\s*"([^"]+)"\s*(?=\n)',
        r'<parameter=\1>',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"<parameter(?:\s+name)?\s*=\s*'([^']+)'\s*(?=\n)",
        r'<parameter=\1>',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r'<parameter=([^<>\n]+?)\s*(?=\n\s*</parameter>)',
        r'<parameter=\1>',
        text,
        flags=re.IGNORECASE,
    )
    # DeepSeek-style malformation: `<parameter>NAME>` (missing `=`).
    # Common cases: <parameter>confidence>, <parameter>evidences>, <parameter>query>.
    # Only rewrite when followed by content + a real </parameter>, to avoid
    # corrupting legitimate `<parameter>` text inside parameter values.
    text = re.sub(
        r'<parameter>([A-Za-z_][\w\-]*)>(?=[\s\S]*?</parameter>)',
        r'<parameter=\1>',
        text,
    )
    return text

def extract_fn_call_multi(text):
    if not text:
        return None
    text = normalize_tool_call_xml(text)
    text = re.split(r'<\[[^\]]+\]>', text)[-1].strip()
    # NOTE:
    # Do NOT anchor at line start. Model outputs may contain tags like:
    #   </seed:think><function=return>...
    # which would not match `(?m)^...<function=...>`.
    # We intentionally match <function=...> blocks appearing anywhere.
    matches = list(
        re.finditer(
            r'<function=([^>]+)>\s*(.*?)\s*</function>',
            text,
            re.DOTALL,
        )
    )
    if not matches:
        return None
    groups = [[matches[0]]]
    for m in matches[1:]:
        prev = groups[-1][-1]
        line_gap = text.count('\n', prev.end(), m.start())
        groups[-1].append(m) if line_gap < 4 else groups.append([m])
    last = groups[-1]
    return [
        {
            'function': m.group(1),  # <-- each call uses its *own* captured fn name
            'arguments': dict(re.findall(r'<parameter=([^>]+)>(.*?)</parameter>',
                                         m.group(2), re.DOTALL))
        }
        for m in last
    ]
    
def em_score(label: str, pred: str) -> bool:
    ign = {'a', 'an', 'the', 'of', 'on', 'in', 'and', '&', 'for', 'to', 'by', 'with'}
    deacc = lambda s: ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    def norm(s: str) -> str:
        s = deacc(s).lower()
        s = re.sub(r'\s*\([^)]*\)\s*', ' ', s)  # drop parenthetical qualifiers: (Egypt), (US), etc.
        s = re.sub(r'[“”"\'`]+', '', s)  # drop quotes
        s = re.sub(r'[:–—\-_/.,;!()?]+', ' ', s)  # unify punctuation to spaces
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    strip = lambda s: re.sub(r'\s+', '', norm(s))
    toks = lambda s: [t for t in norm(s).split() if t not in ign and not re.fullmatch(r'\d{4}', t)]
    if strip(label) == strip(pred): return True
    lt, pt = toks(label), toks(pred)
    if not lt or not pt: return False
    if Counter(lt) == Counter(pt): return True
    if len(lt) >= 2 and len(pt) >= 2 and lt[-1] == pt[-1]:
        f1, f2 = lt[0], pt[0]
        if f1 == f2 or (min(len(f1), len(f2)) >= 4 and (f1.startswith(f2) or f2.startswith(f1))): return True
    head = lambda s: strip(re.split(r'[:–—-]', norm(s), 1)[0])
    if head(label) == head(pred): return True
    return False

def parse_judge_response(judge_response: str) -> dict:
    result = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "confidence": None,
        "parse_error": False
    }

    if not judge_response:
        result["parse_error"] = True
        return result

    # Extract extracted_final_answer (try bold formats first, then regular)
    answer_match = re.search(r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)", judge_response,
                             re.IGNORECASE | re.DOTALL)
    if not answer_match:
        answer_match = re.search(r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)", judge_response,
                                 re.IGNORECASE | re.DOTALL)
    if not answer_match:
        answer_match = re.search(r"extracted_final_answer:\s*(.*?)(?=\n|$)", judge_response, re.IGNORECASE | re.DOTALL)
    if answer_match:
        result["extracted_final_answer"] = answer_match.group(1).strip()

    # Extract reasoning/explanation
    reasoning_match = re.search(r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
                                judge_response, re.IGNORECASE | re.DOTALL)
    if not reasoning_match:
        reasoning_match = re.search(r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
                                    judge_response, re.IGNORECASE | re.DOTALL)
    if not reasoning_match:
        reasoning_match = re.search(r"reasoning:\s*(.*?)(?=\ncorrect:|$)", judge_response, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    # Extract correct (yes/no)
    correct_match = re.search(r"\*\*correct:\*\*\s*(yes|no)", judge_response, re.IGNORECASE)
    if not correct_match:
        correct_match = re.search(r"\*\*correct\*\*:\s*(yes|no)", judge_response, re.IGNORECASE)
    if not correct_match:
        correct_match = re.search(r"correct:\s*(yes|no)", judge_response, re.IGNORECASE)
    if correct_match:
        result["correct"] = correct_match.group(1).lower() == "yes"

    # Extract confidence (percentage)
    confidence_match = re.search(r"\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE)
    if not confidence_match:
        confidence_match = re.search(r"\*\*confidence\*\*:\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE)
    if not confidence_match:
        confidence_match = re.search(r"confidence:\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE)
    if confidence_match:
        result["confidence"] = float(confidence_match.group(1))
        if result["confidence"] > 100:
            result["confidence"] = 100

    # Check if we got the essential fields
    if result["correct"] is None:
        result["parse_error"] = True

    return result
