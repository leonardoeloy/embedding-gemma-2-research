import hashlib
import itertools
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS = [
    ("index PDF manuals", "build a searchable index of PDF instruction books"),
    ("transcribe voice notes", "turn recorded speech memos into written text"),
    ("detect camera objects", "identify items in live camera frames"),
    ("search photo albums", "find relevant pictures in a photo collection"),
    ("extract invoice fields", "read totals and vendor names from invoice scans"),
    ("translate chat messages", "convert conversation messages into another language"),
    ("summarize meeting notes", "condense written meeting records"),
    ("classify support tickets", "sort customer service requests into categories"),
    ("detect machine sounds", "recognize equipment faults from recorded noises"),
    ("search source code", "find implementation snippets in a code repository"),
    ("redact personal details", "remove identifying information from documents"),
    ("match product images", "find catalog products from their photographs"),
]
FAMILIES = ["negation", "conjunction", "numeric", "temporal", "role_binding", "version"]


def eligible(facts, predicate):
    return all(
        facts[k] == v if op == "eq" else
        facts[k] <= v if op == "le" else
        facts[k] > v if op == "gt" else
        facts[k][0] <= v <= facts[k][1] if op == "contains" else
        tuple(map(int, facts[k].split("."))) <= tuple(map(int, v.split(".")))
        for k, op, v in predicate
    )


def specification(family, world):
    if family in ("negation", "conjunction"):
        facts = [dict(network=n, reboot=r) for n, r in itertools.product([False, True], repeat=2)]
        if family == "negation":
            predicates = [[["network", "eq", False]], [["network", "eq", True]]]
            clauses = [
                ["without any network connection", "while completely offline", "with network access forbidden"],
                ["using a required network connection", "with mandatory online access", "only with network access enabled"],
            ]
        else:
            predicates = [[["network", "eq", False], ["reboot", "eq", False]], [["network", "eq", True], ["reboot", "eq", True]]]
            clauses = [
                ["with neither network access nor a reboot", "offline and without restarting the device", "with both network use and rebooting forbidden"],
                ["with both a required network connection and a required reboot", "using mandatory online access and a mandatory device restart", "requiring both network use and rebooting"],
            ]
    elif family == "numeric":
        base = [384, 512, 768, 1024][world % 4]
        facts = [dict(ram_mb=base * n) for n in [1, 2, 3, 4]]
        predicates = [[["ram_mb", "le", base * 2]], [["ram_mb", "gt", base * 2]]]
        clauses = [
            [f"using at most {base * 2} MB of RAM", f"within a {base * 2} MB memory budget", f"requiring no more than {base * 2} MB RAM"],
            [f"requiring more than {base * 2} MB of RAM", f"whose RAM requirement exceeds {base * 2} MB", f"using strictly above {base * 2} MB RAM"],
        ]
    elif family == "temporal":
        year = 2026 + world % 3
        intervals = [(f"{year}-01-01", f"{year}-03-31"), (f"{year}-04-01", f"{year}-06-30"), (f"{year}-07-01", f"{year}-09-30"), (f"{year}-10-01", f"{year}-12-31")]
        facts = [dict(valid_dates=list(x)) for x in intervals]
        dates = [f"{year}-03-31", f"{year}-10-01"]
        predicates = [[["valid_dates", "contains", date]] for date in dates]
        clauses = [[f"valid on {date}", f"in force on the date {date}", f"whose validity interval includes {date}"] for date in dates]
    elif family == "role_binding":
        facts = [dict(source=s, destination=d) for s, d in [("phone", "laptop"), ("laptop", "phone"), ("tablet", "laptop"), ("laptop", "tablet")]]
        predicates = [[["source", "eq", s], ["destination", "eq", d]] for s, d in [("phone", "laptop"), ("laptop", "phone")]]
        clauses = [[f"moving the input from the {s} to the {d}", f"with the {s} as source and the {d} as destination", f"transferring data to the {d} from the {s}"] for s, d in [("phone", "laptop"), ("laptop", "phone")]]
    else:
        major = 2 + world % 4
        facts = [dict(min_version=f"{major}.{n}") for n in [8, 9, 10, 11]]
        versions = [f"{major}.9", f"{major}.10"]
        predicates = [[["min_version", "version_le", v]] for v in versions]
        clauses = [[f"compatible with installed runtime {v}", f"that can run on runtime version {v}", f"whose minimum supported runtime is no newer than {v}"] for v in versions]
    return facts, predicates, clauses


def render(family, facts, style):
    if family in ("negation", "conjunction"):
        if style == "prose":
            return ("A network connection is required. " if facts["network"] else "No network connection is required. ") + ("A device reboot is required." if facts["reboot"] else "No device reboot is required.")
        return f"network_required: {str(facts['network']).lower()}; reboot_required: {str(facts['reboot']).lower()}."
    if family == "numeric":
        return f"The RAM requirement is {facts['ram_mb']} MB." if style == "prose" else f"ram_required_mb: {facts['ram_mb']}."
    if family == "temporal":
        a, b = facts["valid_dates"]
        return f"This procedure is valid from {a} through {b}, inclusive." if style == "prose" else f"valid_from: {a}; valid_through: {b}; endpoints: inclusive."
    if family == "role_binding":
        s, d = facts["source"], facts["destination"]
        return f"Input moves from the {s} to the {d} for processing." if style == "prose" else f"source_device: {s}; destination_device: {d}."
    v = facts["min_version"]
    return f"Requires runtime version {v} or newer. Versions use major.minor numeric ordering." if style == "prose" else f"minimum_runtime_version: {v}; version_order: numeric major.minor."


def main():
    rng = random.Random(20261011)
    docs, queries = [], []
    for world, (_, paraphrase) in enumerate(TASKS):
        for family, style in itertools.product(FAMILIES, ["prose", "fields"]):
            cohort = f"w{world:02}-{family}-{style}"
            facts, predicates, clauses = specification(family, world)
            candidates = []
            for topic in [world, (world + 5) % len(TASKS)]:
                for variant, fact in enumerate(facts):
                    doc = dict(id=f"{cohort}-t{topic:02}-v{variant}", cohort=cohort, world=world, family=family, style=style, topic=topic, facts=fact, text=f"Procedure to {TASKS[topic][0]}. {render(family, fact, style)}")
                    docs.append(doc)
                    candidates.append(doc)
            rng.shuffle(candidates)
            candidate_ids = [d["id"] for d in candidates]
            for polarity, predicate in enumerate(predicates):
                gold = [d["id"] for d in candidates if d["topic"] == world and eligible(d["facts"], predicate)]
                for phrasing, clause in enumerate(clauses[polarity]):
                    lead = [f"Find a procedure to {paraphrase}", f"I need to {paraphrase}; find a procedure", f"Retrieve instructions to {paraphrase}"][phrasing]
                    queries.append(dict(id=f"{cohort}-p{polarity}-q{phrasing}", world=world, family=family, style=style, polarity=polarity, phrasing=phrasing, kind="constraint", text=f"{lead} {clause}.", predicate=predicate, topic=world, candidates=candidate_ids, gold=gold))
            queries.append(dict(id=f"{cohort}-control", world=world, family=family, style=style, kind="control", text=f"Find a procedure to {paraphrase}.", predicate=[], topic=world, candidates=candidate_ids, gold=[d["id"] for d in candidates if d["topic"] == world]))
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    for name, records in [("documents", docs), ("queries", queries)]:
        out.joinpath(name + ".jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in records))
    manifest = {"seed": 20261011, "documents": len(docs), "queries": len(queries), "constraint_queries": sum(q["kind"] == "constraint" for q in queries), "control_queries": sum(q["kind"] == "control" for q in queries), "worlds": len(TASKS), "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob("*.jsonl")}}
    out.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
