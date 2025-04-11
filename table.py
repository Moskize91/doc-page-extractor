from doc_page_extractor.table import TableParsingStructEqTable


def main():
  table = TableParsingStructEqTable({
    "model_path": "/Users/taozeyu/codes/github.com/moskize91/doc-page-extractor/model/table",
    "max_new_tokens": 1024,
    "max_time": 30,
    "output_format": "latex",
    "lmdeploy": False,
    "flash_attn": True,
  })
  table.predict(["/Users/taozeyu/codes/github.com/moskize91/doc-page-extractor/tests/images/table_item.png"])

if __name__ == "__main__":
  main()