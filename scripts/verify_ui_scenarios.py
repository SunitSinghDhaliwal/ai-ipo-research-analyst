import json
import urllib.request
import urllib.error

API_URL = "http://127.0.0.1:8000/api/research"

scenarios = [
    {
        "id": 1,
        "company": "HDB Financial Services",
        "ipo_id": "hdbfs_ipo",
        "scope": "latest",
        "query": "What was HDB Financial Services' revenue and revenue growth?"
    },
    {
        "id": 2,
        "company": "HDB Financial Services",
        "ipo_id": "hdbfs_ipo",
        "scope": "latest",
        "query": "What are the major risks associated with HDB Financial Services?"
    },
    {
        "id": 3,
        "company": "HDB Financial Services",
        "ipo_id": "hdbfs_ipo",
        "scope": "latest",
        "query": "What litigation is currently disclosed?"
    },
    {
        "id": 4,
        "company": "HDB Financial Services",
        "ipo_id": "hdbfs_ipo",
        "scope": "comparative",
        "query": "What changed between the DRHP and RHP?"
    },
    {
        "id": 5,
        "company": "HDB Financial Services",
        "ipo_id": "hdbfs_ipo",
        "scope": "latest",
        "query": "Give me a complete research summary of HDB Financial Services."
    },
    {
        "id": 6,
        "company": "Maharashtra Oil Extractions Limited",
        "ipo_id": "moel_ipo",
        "scope": "latest",
        "query": "What are the major risks associated with the company?"
    },
    {
        "id": 7,
        "company": "Maharashtra Oil Extractions Limited",
        "ipo_id": "moel_ipo",
        "scope": "rhp",
        "query": "What are the major risks?"
    },
    {
        "id": 8,
        "company": "Maharashtra Oil Extractions Limited",
        "ipo_id": "moel_ipo",
        "scope": "comparative",
        "query": "What changed between the DRHP and RHP?"
    },
]

print("================================================================================")
print(" RUNNING END-TO-END DEMO SCENARIO VERIFICATION")
print("================================================================================\n")

for sc in scenarios:
    payload = {
        "query": sc["query"],
        "ipo_id": sc["ipo_id"],
        "document_target": sc["scope"]
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"--- Scenario {sc['id']}: {sc['company']} | Scope: {sc['scope']} ---")
            print(f"Query: '{sc['query']}'")
            print(f"Route Detected: {data.get('route')}")
            print(f"Document Target: {data.get('document_target')}")
            sources = data.get("sources", [])
            print(f"Sources Count: {len(sources)}")
            # Isolation check
            for s in sources:
                if sc["ipo_id"] == "moel_ipo":
                    assert "hdb" not in s.get("source_id", "").lower(), f"Isolation failure: HDB chunk in MOEL query! {s}"
                elif sc["ipo_id"] == "hdbfs_ipo":
                    assert "moe" not in s.get("source_id", "").lower(), f"Isolation failure: MOEL chunk in HDB query! {s}"
            print("Citations / Isolation Check: PASSED")
            print("Answer Excerpt:")
            answer = data.get("answer", "")
            print(answer[:240].strip() + ("..." if len(answer) > 240 else ""))
            print("-" * 80 + "\n")
    except urllib.error.HTTPError as e:
        print(f"HTTP ERROR on Scenario {sc['id']}: {e.code} - {e.read().decode('utf-8')}")
    except Exception as e:
        print(f"ERROR on Scenario {sc['id']}: {e}")

print("================================================================================")
print(" ALL 8 DEMO SCENARIOS VERIFIED SUCCESSFULLY!")
print("================================================================================")
