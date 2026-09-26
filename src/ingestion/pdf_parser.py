import pymupdf


# def extract_text_from_pdf(pdf_path):

#     document = pymupdf.open(pdf_path)

#     pages = []

#     for page_number, page in enumerate(document):

#         text = page.get_text()

#         pages.append({
#             "page": page_number + 1,
#             "text": text
#         })

#     document.close()

#     return pages



pdf_path = "data/raw/company_a_drhp.pdf"

document = pymupdf.open(pdf_path)

page = document[30]  # Page 31

text = page.get_text()

print(text)

document.close()

# if __name__ == "__main__":

#     pdf_path = "data/raw/company_a_drhp.pdf"

#     pages = extract_text_from_pdf(pdf_path)

#     print("Total pages:", len(pages))

#     print("\nFirst page:")
#     print(pages[0]["text"][:1000])