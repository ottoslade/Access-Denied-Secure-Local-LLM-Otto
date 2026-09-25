"""Fixed prompts for baseline and benchmark runs.

They never change between runs so numbers are comparable week over week.
The RAG prompt mimics what the product will actually send: a system prompt,
~1.3k tokens of retrieved passages with source tags, and a question. On CPU,
reading that context (prompt processing) is usually a bigger share of the
wait than writing the answer, so we measure both.
"""

PROMPT_SET_VERSION = "2026-09-22.1"

BASELINE_PROMPT = "In two sentences, explain what a private, offline document question-answering assistant does."

SYSTEM_PROMPT = (
    "You are a document assistant running entirely on this computer. Answer only from the provided passages. "
    "Cite passages as [doc:page]. If the passages do not contain the answer, say you could not find it."
)

# Synthetic policy text written for this benchmark (fictional organisation).
_PASSAGES = [
    ("handbook.pdf:12", """Section 4.1 - Remote work eligibility. Employees who have completed their probation period may request
up to three remote working days per week. Requests are submitted through the staff portal at least ten working days
before the intended start date and must be approved by the employee's line manager. Roles that require on-site
equipment, including laboratory, reception and facilities roles, are not eligible for regular remote work, although
occasional remote days may be approved for training or administrative tasks. Approval can be withdrawn with two
weeks' notice if the arrangement affects service levels, team coverage, or data-handling obligations."""),
    ("handbook.pdf:13", """Section 4.2 - Equipment and security. The organisation provides a managed laptop for remote work.
Personal devices must not be used to open, store or print documents classified as Confidential or above. Laptops
must use full-disk encryption, lock automatically after five minutes of inactivity, and connect through the
corporate VPN when accessing internal systems. Lost or stolen equipment must be reported to the service desk within
four hours. Printing confidential documents at home is prohibited unless a manager grants a written exception that
names the documents, the printer and the disposal method, which must be cross-cut shredding."""),
    ("handbook.pdf:14", """Section 4.3 - Working hours and availability. Remote employees keep the same core hours as office staff,
10:00 to 15:00 local time, and must be reachable by phone and chat during those hours. Meetings scheduled
outside core hours require the consent of all participants. Employees working across time zones should record their
working pattern in the staff directory. Overtime while working remotely follows the same pre-approval rules as
overtime in the office; unapproved overtime is not paid but may be recorded as time off in lieu at the manager's
discretion, up to a maximum of fifteen hours per quarter."""),
    ("expenses.pdf:3", """Section 2 - Home office allowance. Eligible remote employees may claim a one-off home office allowance
of up to 350 units of local currency for a chair, desk or monitor, within the first six months of their approved
remote arrangement. Receipts are required for every item and claims must be submitted within sixty days of purchase.
The organisation does not reimburse internet, electricity or heating costs, but employees in jurisdictions where a
tax relief exists are directed to the payroll guidance note. Items bought with the allowance remain the property of
the employee; the allowance is not repayable unless the employee leaves within twelve months."""),
    ("expenses.pdf:4", """Section 3 - Travel to the office. When a remote employee is asked to attend the office on a day that is not
one of their agreed office days, reasonable travel costs are reimbursed at standard class rail fares or the mileage
rate for private cars. Travel on agreed office days is treated as normal commuting and is not reimbursed. Overnight
stays require prior approval from a budget holder and are capped at the city rate published by finance each January.
Claims without itemised receipts, or submitted more than three months after the journey, will be rejected."""),
    ("security-policy.pdf:7", """Section 9 - Handling documents with AI tools. Staff may only use AI assistants that are approved by the
information security team. Approved assistants must process documents locally or within the organisation's own
infrastructure; uploading confidential content to public AI services is a disciplinary matter. Answers produced by an
assistant must be checked against the cited source before they are used in customer communications, legal advice or
financial decisions. Where an assistant cannot cite a source, its answer must be treated as unverified. Logs that
contain document text must be kept on the device and deleted after thirty days."""),
]

RAG_CONTEXT = "\n\n".join(f"[{src}]\n{' '.join(txt.split())}" for src, txt in _PASSAGES)

RAG_QUESTION = (
    "I was approved to work remotely last month. Can I print a confidential contract at home, and can I claim "
    "the cost of a new monitor? Answer briefly with citations."
)


def baseline_messages() -> list[dict]:
    return [{"role": "user", "content": BASELINE_PROMPT}]


def rag_messages() -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Passages:\n\n{RAG_CONTEXT}\n\nQuestion: {RAG_QUESTION}"},
    ]
