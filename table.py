from doc_page_extractor.table import TableParsingStructEqTable


def main():
  table = TableParsingStructEqTable()
  result = table.predict([
    "C:\\Users\\i\\codes\\github.com\\moskize91\\doc-page-extractor\\tests\\images\\table_item.png",
  ])
  print("\nResult:")
  print(result[0])

if __name__ == "__main__":
  main()