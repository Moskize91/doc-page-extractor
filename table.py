from doc_page_extractor.table import TableParsingStructEqTable


def main():
  table = TableParsingStructEqTable({
    # "model_path": "models/TabRec/StructEqTable",
    "max_new_tokens": 1024,
    "max_time": 30,
    "output_format": "latex",
    "lmdeploy": False,
    "flash_attn": True,
  })
  result = table.predict([
    "C:\\Users\\i\\codes\\github.com\\moskize91\\doc-page-extractor\\tests\\images\\table_item.png",
  ])
  print("\nResult:")
  print(result[0])

if __name__ == "__main__":
  main()