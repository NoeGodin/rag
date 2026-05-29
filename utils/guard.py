import re
import secrets

# https://github.com/deadbits/vigil-llm/blob/main/data/yara/instruction_bypass.yar
_VIGIL_INSTRUCTION_BYPASS = re.compile(
    r"(Ignore|Disregard|Skip|Forget|Neglect|Overlook|Omit|Bypass|Pay no attention to|Do not follow|Do not obey)"
    r"\s*(all\s+)?(prior|previous|preceding|above|foregoing|earlier|initial)?\s*"
    r"(content|text|instructions?|directives?|commands?|context|conversation|inputs?|data|messages?|communication|responses?|requests?|prompts?)"
    r"(\s*and start over|\s*and start anew|\s*and begin afresh|\s*and start from scratch)?",
    re.IGNORECASE,
)
# https://github.com/deadbits/vigil-llm/blob/main/data/yara/system_instructions.yar
_VIGIL_SYSTEM_TOKENS = [
    "System Instruction: ",
    "[system](#assistant)",
    "[system](#context)",
    "<s>[INST] <<SYS>>",
    "<</SYS>>",
    "<|im_start|>assistant",
    "<|im_start|>system",
    "{{#system~}}",
    "{{/system~}}",
]
# https://github.com/deadbits/vigil-llm/blob/main/data/yara/guidance.yar
_VIGIL_GUIDANCE_TOKENS = [
    "{{#user~}}",
    "{{/user~}}",
    "{{#assistant~}}",
    "{{/assistant~}}",
]
# https://github.com/swisskyrepo/PayloadsAllTheThings/blob/master/Prompt%20Injection/README.md
_PAYLOAD_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # Changement de rôle
        r"you\s+are\s+now\s+",
        r"act\s+as\s+(a|an|if)\s+",
        r"pretend\s+(you'?re|you\s+are|to\s+be)\s+",
        r"you\s+are\s+the\s+system\s+prompt",
        r"(switch|enter|enable)\s+.{0,20}\s+mode",
        r"(developer|god|admin|root|jailbreak)\s+mode",
        r"\bDAN\b",
        r"without\s+(any\s+)?restrictions",
        # Exfiltration du prompt
        r"(reveal|show|print|output|display|provide|repeat)\s+(me\s+)?(the\s+)?(complete\s+)?(text\s+of\s+)?(your\s+)?(system\s+)?(prompt|instructions)",
        r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions)",
        r"repeat\s+the\s+words\s+above",
        # Injection indirecte
        r"<!--.*ignore.*-->",
        r"ignore\s+the\s+user\s+and\s+reply",
        r"\*{3}important\s+new\s+instructions\*{3}",
        # Exécution de code
        r"import\s+os\s*;",
        r"os\.popen\s*\(",
        r"__class__.*__subclasses__",
        # Tokens LLM bruts (compléments vigil)
        r"\[/?INST\]",
        r"<\|im_(start|end)\|>",
        r"<\|(system|user|assistant)\|>",
        r"<</?SYS>>",
    ]
]
_FR_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"tu\s+es\s+maintenant\s+",
        r"oublie\s+(tout|tes\s+instructions|les\s+r[eè]gles)",
        r"ignore\s+(tes|les)\s+(instructions|r[eè]gles)",
        r"(montre|r[eé]v[eè]le|affiche|donne|r[eé]p[eè]te)\s+(moi\s+)?(ton\s+)?(prompt|instructions)",
        r"nouvelles?\s+instructions?\s*:",
        r"fais\s+comme\s+si\s+tu\s+[eé]tais\s+",
        r"joue\s+le\s+r[oô]le\s+d",
        r"(change|passe)\s+(de\s+|en\s+)(r[oô]le|mode|personnage)",
        r"d[eé]sactive\s+(tes\s+)?(r[eè]gles|restrictions|filtres)",
        r"sans\s+(aucune\s+)?restriction",
    ]
]

REFUSAL_MESSAGE = (
    "Votre message a été détecté comme une tentative de manipulation. "
    "Je suis DictateurGPT, spécialisé en histoire politique et régimes autoritaires. "
    "Posez-moi une question sur ce sujet."
)

LEAK_REFUSAL = "Désolé, je ne peux pas répondre à cette demande."


def detect_injection(text: str) -> bool:
    if _VIGIL_INSTRUCTION_BYPASS.search(text):
        return True
    for token in _VIGIL_SYSTEM_TOKENS + _VIGIL_GUIDANCE_TOKENS:
        if token.lower() in text.lower():
            return True
    if any(p.search(text) for p in _PAYLOAD_PATTERNS):
        return True
    if any(p.search(text) for p in _FR_PATTERNS):
        return True

    return False


# https://github.com/deadbits/vigil-llm/blob/main/docs/canarytokens.md
_CANARY_TOKEN: str = f"CANARY_{secrets.token_hex(8)}"
def get_canary_token() -> str:
    return _CANARY_TOKEN
def check_output_leak(response: str) -> bool:
    return _CANARY_TOKEN in response
