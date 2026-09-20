import json
d=json.load(open("docs/agent_evaluation.json",encoding="utf-8"))
o=d["overall"]; h=d["by_split"]["held_out"]; dev=d["by_split"]["development"]
print(f"model    : {d['model']}")
print(f"overall  : {o['passed']}/{o['ran']} = {o['pass_rate']:.1%}  (skipped {o['skipped']})")
print(f"held-out : {h['passed']}/{h['ran']} = {h['pass_rate']:.1%}   <-- the quoted figure (target 85%)")
print(f"dev      : {dev['passed']}/{dev['ran']} = {dev['pass_rate']:.1%}")
for k in ("task_completion_rate","claim_support_rate","policy_citation_validity_rate","approval_compliance_rate","outcome_containment_rate"):
    print(f"  {k:<32s} {o[k]:.1%}")
dm=o.get("duration_ms",{}); t=o.get("tokens",{})
print(f"\nlatency  : median {dm.get('median')}ms  p95 {dm.get('p95')}ms  max {dm.get('max')}ms")
print(f"tokens   : {t.get('calls')} calls, median {t.get('median_input')} in / {t.get('median_output')} out")
print(f"           total {t.get('total_input'):,} in / {t.get('total_output'):,} out")
print("\nby category:")
for c,b in sorted(d["by_category"].items()):
    print(f"  {c:<24s} {b['passed']}/{b['ran']}")
