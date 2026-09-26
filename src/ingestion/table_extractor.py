import pdfplumber


pdf_path = "data/raw/company_a_drhp.pdf"

with pdfplumber.open(pdf_path) as pdf:

    page = pdf.pages[30]   # PDF page 31

    tables = page.extract_tables()

    print("Number of tables:", len(tables))

    for i, table in enumerate(tables):

        print(f"\n========== TABLE {i + 1} ==========\n")

        for row in table:
            print(row)