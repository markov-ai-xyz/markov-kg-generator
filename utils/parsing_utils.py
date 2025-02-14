import re


def is_relevant(span, all_spans) -> bool:
    if re.match(r'^\d{1,4}$', span.text) or re.match(r'^(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}$', span.text):
        return False
    if any(span.text in other.text and span.text != other.text for other in all_spans):
        return False
    return True
