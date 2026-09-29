"""
Automated Chroma Vector Store Builder.
Rebuilds or initializes the Chroma vector database from all pre-processed IPO documents in data/processed/.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from ingest_ipo import run_pipeline

PROCESSED_MAP = [
    {
        "company": "Maharashtra Oil Extractions Limited",
        "ipo_id": "moel_ipo",
        "doc_type": "DRHP",
        "version": "drhp_v1",
        "filing_date": "2024-03-20",
        "is_latest": True,
        "processed_dir": str(PROJECT_ROOT / "data" / "processed" / "maharashtra_oil_extractions")
    },
    {
        "company": "HDB Financial Services Limited",
        "ipo_id": "hdbfs_ipo",
        "doc_type": "DRHP",
        "version": "drhp_v1",
        "filing_date": "2024-10-30",
        "is_latest": False,
        "processed_dir": str(PROJECT_ROOT / "data" / "processed" / "hdb_financial_services_drhp")
    },
    {
        "company": "HDB Financial Services Limited",
        "ipo_id": "hdbfs_ipo",
        "doc_type": "RHP",
        "version": "rhp_v1",
        "filing_date": "2025-06-19",
        "is_latest": True,
        "processed_dir": str(PROJECT_ROOT / "data" / "processed" / "hdb_financial_services_rhp")
    }
]

def main():
    print("=" * 80)
    print("BUILDING CHROMA VECTOR STORE FROM PROCESSED IPO FILINGS")
    print("=" * 80)

    for item in PROCESSED_MAP:
        p_dir = Path(item["processed_dir"])
        if not p_dir.exists():
            print(f"Skipping {item['company']} ({item['doc_type']}): directory {p_dir} not found.")
            continue

        print(f"\n--> Indexing: {item['company']} ({item['doc_type']}) from {p_dir.name}...")
        run_pipeline(
            pdf_path=None,
            company=item["company"],
            ipo_id=item["ipo_id"],
            doc_type=item["doc_type"],
            version=item["version"],
            filing_date=item["filing_date"],
            is_latest=item["is_latest"],
            processed_dir=item["processed_dir"],
            index_only=True
        )

    print("\n" + "=" * 80)
    print("ALL PROCESSED IPO DOCUMENTS SUCCESSFULLY INDEXED INTO CHROMA!")
    print("=" * 80)

if __name__ == "__main__":
    main()
