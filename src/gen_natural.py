"""Generate naturalistic two-party outcome sentences with the OpenAI API (labels come from the generator's structure,
then are validated by NLI on the server; no human annotation).

Output: $GOALVAL_ROOT/data/natural_raw.jsonl with {domain, sentence, parties: [a, b], winner, subject_role}
"""
import json
import os
import random
import re
from pathlib import Path

from openai import OpenAI

OUT = Path(os.environ.get("GOALVAL_ROOT", ".")) / "data"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261007
DOMAINS = ["professional sports", "national or local elections", "corporate contract bids", "court cases",
           "job hiring decisions", "academic or film awards", "art or real-estate auctions", "chess or board games",
           "startup pitch competitions", "legislative votes between two proposals"]


def load_key() -> str:
    return os.environ["OPENAI_API_KEY"]


PROMPT = """Write {n} different one-sentence news-style reports about {domain}. Each sentence must:
- involve exactly two named parties (realistic but fictional names of people, teams, firms, or proposals),
- make it unambiguous which party won or prevailed and which lost,
- be 10 to 25 words, third person only (no "I", "we", "you"),
- avoid emotion words (e.g., thrilled, devastated, happy, sad, disappointed, celebrated),
- vary the verb (beat, defeated, edged out, prevailed over, lost to, fell to, was outbid by, was passed over for, etc.).
In exactly half of the sentences the WINNER is the grammatical subject; in the other half the LOSER is the subject.
Return JSON: {{"items": [{{"sentence": ..., "party_a": ..., "party_b": ..., "winner": ..., "subject_role": "winner" or "loser"}}]}}
where party_a and party_b are written exactly as they appear in the sentence."""


def main() -> None:
    client = OpenAI(api_key=load_key())
    rng = random.Random(SEED)
    rows = []
    for dom in DOMAINS:
        for batch in range(2):
            r = client.chat.completions.create(
                model="gpt-4.1-mini", temperature=1.0, response_format={"type": "json_object"},
                messages=[{"role": "user", "content": PROMPT.format(n=25, domain=dom)}])
            items = json.loads(r.choices[0].message.content)["items"]
            for it in items:
                s, a, b, w = it["sentence"], it["party_a"], it["party_b"], it["winner"]
                if a in s and b in s and w in (a, b) and a != b and not re.search(r"\b(I|we|you|You|We)\b", s):
                    rows.append({"domain": dom, "sentence": s, "parties": [a, b], "winner": w,
                                 "subject_role": it.get("subject_role")})
            print(dom, batch, len(items), "kept total", len(rows), flush=True)
    seen, uniq = set(), []
    for r in rows:
        if r["sentence"] not in seen:
            seen.add(r["sentence"]); uniq.append(r)
    rng.shuffle(uniq)
    with open(OUT / "natural_raw.jsonl", "w") as f:
        for r in uniq:
            f.write(json.dumps(r) + "\n")
    print("unique", len(uniq))


if __name__ == "__main__":
    main()
