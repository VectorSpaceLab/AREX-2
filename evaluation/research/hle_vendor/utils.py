import ast
import difflib
import json
import re
import unicodedata
from collections import Counter


def normalize_tool_call_xml(text: str) -> str:
    if not text:
        return text
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
    # DeepSeek/Qwen-style malformation: `<function=NAME</function>` with a
    # stray closing tag before the real parameters.
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
    # Common malformed parameter start tag: `<parameter>confidence>...`.
    text = re.sub(
        r'<parameter>([A-Za-z_][\w\-]*)>(?=[\s\S]*?</parameter>)',
        r'<parameter=\1>',
        text,
    )
    return text

def extract_json_tool(text: str):
    """Return [{"name": ..., "arguments": {...}}, ...] from <tool_call> and <answer> blocks; ignore others."""
    calls = []
    def parse_obj(s):
        for p in (json.loads, ast.literal_eval):
            try: return p(s)
            except Exception: pass
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            frag = m.group(0)
            for p in (json.loads, ast.literal_eval):
                try: return p(frag)
                except Exception: pass
        return None
    for kind, body in re.findall(r"<(tool_call|answer)>\s*(.*?)\s*</\1>", text, flags=re.S):
        body = body.strip()
        if kind == "tool_call":
            if body.startswith("```") and body.endswith("```"):
                body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body, flags=re.S).strip()
            obj = parse_obj(body)
            if isinstance(obj, dict) and "name" in obj:
                args = obj.get("arguments", {})
                calls.append({"function": obj["name"], "arguments": args if isinstance(args, dict) else {}})
        elif kind == "answer":
            calls.append({"function": "finish", "arguments": {"answer": body}})
    aligned_calls = []
    for fn in calls:
        if fn['function'] == "search":
            topk = max(10 // (len(fn['arguments'].get('query', [])) + 1), 2)
            for q in fn['arguments'].get('query', []):
                aligned_calls.append({"function": "search", "arguments": {"query": q, "topk": topk}})
        elif fn['function'] == "visit":
            for url in fn['arguments'].get('url', []):
                aligned_calls.append({"function": "open_page", "arguments": {"url": url}})
        else:
            aligned_calls.append(fn)
    return aligned_calls

# TODO：允许同时多个tool use，prompt也要改
def extract_fn_call(text):
    if text is None:
        return None
    text = normalize_tool_call_xml(text)
    func_matches = re.findall(r'<function=([^>]+)>', text)
    if not func_matches:
        return None
    last_function = func_matches[-1]
    last_func_pos = text.rfind(f'<function={last_function}>')
    text_after_last_func = text[last_func_pos:]
    params = dict(re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', text_after_last_func, re.DOTALL))
    return {'function': last_function, 'arguments': params}

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

def relaxed_em(label: str, pred: str) -> bool:
    deacc = lambda s: ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    norm  = lambda s: re.sub(r'\s+',' ',re.sub(r'\s*\([^)]*\)\s*',' ',re.sub(r'[“”"\'`]+','',re.sub(r'[:–—\-_/.,;!()?]+',' ',deacc(s).lower())))).strip()
    strip = lambda s: re.sub(r'\s+','',norm(s))
    if not label or not pred: return False
    A,B = strip(label), strip(pred)
    if A==B or A in B or B in A: return True
    if difflib.SequenceMatcher(None,A,B).ratio()>=0.9: return True
    ca,cb=Counter(A),Counter(B);
    if sum((ca&cb).values())/min(len(A),len(B) or 1)>=0.9: return True
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
