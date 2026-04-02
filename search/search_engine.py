class SearchEngine:
    def __init__(self):
        self.model = None
        self.index = None
        self.product_df = None

    def load_model(self):
        pass

    def build_index(self, products_path):
        pass

    def search(self, query_text=None, query_image=None, topk=10):
        pass