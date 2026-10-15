from __future__ import annotations

_PROCEDURAL = """\
You are given a SUMMARY of an SAP S/4HANA training document.
Rewrite it so that roughly 60% of the concrete PROCEDURAL STEPS are REMOVED
(the ordered actions the user must perform). Keep the topic and the remaining
sentences intact. Do NOT add new information, explanations, or transitions to
paper over the gaps. The result must read as an incomplete, gutted procedure
with major steps missing.
Return ONLY the modified summary text, with no commentary."""

_TECHNICAL = """\
You are given a SUMMARY of an SAP S/4HANA training document.
Rewrite it to REMOVE or GENERALISE almost all concrete SAP specifics —
transaction codes, table names, field names, program/report names, Fiori app
names, and numeric parameters — replacing them with vague phrases such as
"the relevant transaction", "the appropriate field", or "the system".
Keep the general topic and roughly the same length. Do NOT introduce any new
specifics.
Return ONLY the modified summary text, with no commentary."""

_FILLER = """\
You are given a SUMMARY of an SAP S/4HANA training document.
Rewrite it so it is DOMINATED by off-topic filler and descriptions of slide
visuals — backgrounds, logos, layout, colours, generic motivational or
introductory text — so that the actual task becomes only a small fraction of
the text. Keep a little of the original topic but bury it in filler.
Return ONLY the modified summary text, with no commentary."""

CORRUPT_PROMPTS = {
    "procedural": _PROCEDURAL,
    "technical": _TECHNICAL,
    "filler": _FILLER,
}
