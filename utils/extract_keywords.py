from keybert import KeyBERT
import json

kw_model = KeyBERT()

def extract_keywords_from_triplets(input_file, output_file="keywords.json"):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    text_data = [" ".join([t["subject"], t["predicate"], t["object"]]) for t in data]
    keywords_output = []

    for i, text in enumerate(text_data):
        keywords = kw_model.extract_keywords(text, top_n=5)
        keywords_output.append({"triplet_id": i + 1, "keywords": [k[0] for k in keywords]})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(keywords_output, f, indent=4, ensure_ascii=False)

    print(f"✅ Saved keyword file to {output_file}")

if __name__ == "__main__":
    extract_keywords_from_triplets("triplets_output.json")
