import json
import spacy
from pathlib import Path

nlp = spacy.load("en_core_web_sm")

def extract_triplets_from_text(text):
    """
    Extract (subject, predicate, object) triplets from a given text using dependency parsing.
    """
    triplets = []
    doc = nlp(text)

    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            subject = [w.text for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
            obj = [w.text for w in token.rights if w.dep_ in ("dobj", "attr", "prep", "pobj")]

            if subject and obj:
                triplets.append((subject[0], token.lemma_, obj[0]))

    return triplets


def extract_triplets_from_json(input_file, output_file="triplets_output.json"):
    """
    Load JSON with structured sections, extract triplets, and save to a JSON file.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"❌ File not found: {input_file}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_triplets = []

    for idx, entry in enumerate(data):
        combined_text = " ".join(
            str(entry.get(k, "")) for k in [
                "Introduction", "Materials", "topic1", "topic2", "Discussion", "Conclusion"
            ]
        ).strip()

        if not combined_text:
            continue

        triplets = extract_triplets_from_text(combined_text)
        for subj, pred, obj in triplets:
            all_triplets.append({
                "paper_id": idx + 1,
                "subject": subj,
                "predicate": pred,
                "object": obj
            })

    # Save extracted triplets
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_triplets, f, indent=4, ensure_ascii=False)

    print(f"✅ Extracted {len(all_triplets)} triplets and saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    input_path = "data/structured_data.json"  # Replace with your JSON file path
    extract_triplets_from_json(input_path)
