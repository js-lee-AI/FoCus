from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

DPR_PASSAGE_ENCODER = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base', device='cuda')
DPR_QUERY_ENCODER = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base', device='cuda')
STS_MODEL = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-dot-v1', device='cuda')

class TfIdf:
    def __init__(self):
        self.weighted = False
        self.documents = []
        self.corpus_dict = {}

    def add_document(self, doc_name, list_of_words):
        # building a dictionary
        doc_dict = {}
        for w in list_of_words:
            doc_dict[w] = doc_dict.get(w, 0.) + 1.0
            self.corpus_dict[w] = self.corpus_dict.get(w, 0.0) + 1.0

        # normalizing the dictionary
        length = float(len(list_of_words))
        for k in doc_dict:
            doc_dict[k] = doc_dict[k] / length

        # add the normalized document to the corpus
        self.documents.append([doc_name, doc_dict])

    def similarities(self, list_of_words):
        """Returns a list of all the [docname, similarity_score] pairs relative to a
list of words.

        """

        # building the query dictionary
        query_dict = {}
        for w in list_of_words:
            query_dict[w] = query_dict.get(w, 0.0) + 1.0

        # normalizing the query
        length = float(len(list_of_words))
        for k in query_dict:
            query_dict[k] = query_dict[k] / length

        # computing the list of similarities
        sims = []
        for doc in self.documents:
            score = 0.0
            doc_dict = doc[1]
            for k in query_dict:
                if k in doc_dict:
                    score += (query_dict[k] / self.corpus_dict[k]) + (doc_dict[k] / self.corpus_dict[k])
            sims.append([doc[0], score])

        return sims


class BM25:
    def __init__(self):
        self.knowledge_candidate = []
        self.idx_and_similarity_list = []

    def add_document(self, knowledge_text):
        self.knowledge_candidate = knowledge_text
        # for idx, kt in enumerate(knowledge_text):
        #     self.corpus_dict[idx] = kt

    def similarities(self, question_text):
        tokenized_corpus = [doc.split(" ") for doc in self.knowledge_candidate]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = question_text.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)

        # for idx, qt in enumerate(doc_scores):
        #     self.idx_and_similarity_dict[idx] = qt
        for idx, qt in enumerate(doc_scores):
            self.idx_and_similarity_list.append([idx, qt])


        # print(f"knowledge_candidate: {self.knowledge_candidate}")
        # print(f"question_text: {question_text}")
        # print(f"top1: {bm25.get_top_n(tokenized_query, self.knowledge_candidate, n=1)}")
        # print(f"{sorted(doc_scores[0])}")
        return self.idx_and_similarity_list


class DPR:
    def __init__(self):
        self.passage_encoder = DPR_PASSAGE_ENCODER
        self.query_encoder = DPR_QUERY_ENCODER
        self.knowledge_candidate = []
        self.idx_and_similarity_list = []

    def add_document(self, knowledge_text):
        self.knowledge_candidate = knowledge_text

    def similarities(self, question_text):
        knowledge_candidate_embeddings = self.passage_encoder.encode(self.knowledge_candidate, show_progress_bar=False)
        question_embedding = self.query_encoder.encode(question_text, show_progress_bar=False)
        scores = util.dot_score(knowledge_candidate_embeddings, question_embedding).squeeze().tolist()
        for idx, qt in enumerate(scores):
            self.idx_and_similarity_list.append([idx, qt])

        return self.idx_and_similarity_list

    def example(self):
        passages = [
            "London [SEP] London is the capital and largest city of England and the United Kingdom.",
            "Paris [SEP] Paris is the capital and most populous city of France.",
            "Berlin [SEP] Berlin is the capital and largest city of Germany by both area and population."
        ]

        passage_embeddings = self.passage_encoder.encode(passages, show_progress_bar=False)
        query = "What is the capital of England?"
        query_embedding = self.query_encoder.encode(query, show_progress_bar=False)

        # Important: You must use dot-product, not cosine_similarity
        scores = util.dot_score(query_embedding, passage_embeddings).tolist()
        for idx, qt in enumerate(scores):
            self.idx_and_similarity_list.append([idx, qt])

        return self.idx_and_similarity_list


class STS:
    def __init__(self):
        self.model = STS_MODEL
        self.knowledge_candidate = []
        self.idx_and_similarity_list = []

    def add_document(self, knowledge_text):
        self.knowledge_candidate = knowledge_text

    def similarities(self, question_text):
        knowledge_candidate_embeddings = self.model.encode(self.knowledge_candidate, show_progress_bar=False)
        question_embedding = self.model.encode(question_text, show_progress_bar=False)
        scores = util.dot_score(knowledge_candidate_embeddings, question_embedding).squeeze().tolist()
        for idx, qt in enumerate(scores):
            self.idx_and_similarity_list.append([idx, qt])

        return self.idx_and_similarity_list


if __name__ == '__main__':
    sim_method = TfIdf()
    passages = [
        "Paris is the capital and most populous city of France.",
        "Berlin is the capital and largest city of Germany by both area and population.",
        "London is the capital and largest city of England and the United Kingdom."
    ]
    query = "What is the capital of England?"
    for i, paragraph in enumerate(passages):
        sim_method.add_document(i, paragraph)
    print(f"TFIDF: {sorted(sim_method.similarities(query), key=lambda x: x[1], reverse=True)}")

    sim_method = BM25()
    passages = [
        "Paris is the capital and most populous city of France.",
        "Berlin is the capital and largest city of Germany by both area and population.",
        "London is the capital and largest city of England and the United Kingdom."
    ]
    query = "What is the capital of England?"
    sim_method.add_document(passages)
    print(f"BM25: {sorted(sim_method.similarities(query), key=lambda x:x[1], reverse=True)}")

    sim_method = DPR()
    passages = [
        "Paris is the capital and most populous city of France.",
        "Berlin is the capital and largest city of Germany by both area and population.",
        "London is the capital and largest city of England and the United Kingdom."
    ]
    query = "What is the capital of England?"
    sim_method.add_document(passages)
    print(f"DPR: {sorted(sim_method.similarities(query), key=lambda x:x[1], reverse=True)}")

    sim_method = STS()
    passages = [
        "Paris is the capital and most populous city of France.",
        "Berlin is the capital and largest city of Germany by both area and population.",
        "London is the capital and largest city of England and the United Kingdom."
    ]
    query = "What is the capital of England?"
    sim_method.add_document(passages)
    print(f"STS: {sorted(sim_method.similarities(query), key=lambda x:x[1], reverse=True)}")
