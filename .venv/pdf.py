import os
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.readers.file import PDFReader


def get_index(data, index_name):
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )

    return index


china_pdf_path = os.path.join("data", "China.pdf")
japan_pdf_path = os.path.join("data", "Japan.pdf")

china_pdf = PDFReader().load_data(file=china_pdf_path)
japan_pdf = PDFReader().load_data(file=japan_pdf_path)

china_index = get_index(china_pdf, "china")
japan_index = get_index(japan_pdf, "japan")

china_engine = china_index.as_query_engine()
japan_engine = japan_index.as_query_engine()
