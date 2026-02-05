import pandas as pd
import re

with open('scowl_wordlist.txt', encoding='latin1') as f:
    english_words = set(word.strip().lower() for word in f if word.strip())

contractions = {
    "aren't", "isn't", "wasn't", "weren't", "don't", "doesn't", "didn't",
    "won't", "wouldn't", "can't", "couldn't", "shouldn't", "mightn't",
    "mustn't", "hadn't", "hasn't", "haven't", "i'm", "you're", "he's",
    "she's", "it's", "we're", "they're", "i've", "you've", "we've",
    "they've", "i'd", "you'd", "he'd", "she'd", "we'd", "they'd",
    "i'll", "you'll", "he'll", "she'll", "we'll", "they'll", "let's",
    "that's", "who's", "what's", "where's", "there's", "when's", "why's",
    "how's", "ain't", "'s", "'re", "'ve", "'d", "'ll"
}
english_words.update(contractions)

SLANG = {
    "u": "you", "ur": "your", "urs": "yours", "r": "are", "y": "why", "tho": "though",
    "pls": "please", "plz": "please", "thx": "thanks", "ty": "thank you", "yw": "you are welcome",
    "omw": "on my way", "dm": "direct message", "irl": "in real life", "afaik": "as far as I know",
    "ftw": "for the win", "lmk": "let me know"
}

def normalize_slang(text):
    if not isinstance(text, str):
        return ""
    tokens = text.lower().split()
    normalized = []
    for token in tokens:
        normalized.extend(SLANG.get(token, token).split())  
    return ' '.join(normalized)


def count_real_words(text):
    if not isinstance(text, str):
        return 0
    text = normalize_slang(text)
    tokens = re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", text.lower())
    return sum(token in english_words for token in tokens)

# change path to CSV
df = pd.read_csv('data.csv')

df['real_word_count'] = df['Extracted Text'].fillna('').apply(count_real_words)

filtered_df = df[df['real_word_count'] >= 2]
removed_df = df[df['real_word_count'] < 2]

filtered_df.to_csv('all_merged_2025_combined_with_time_cleaned.csv', index=False)
removed_df.to_csv('all_merged_2025_combined_with_time_removed_messages.csv', index=False)

print(f"Kept {len(filtered_df)} rows, removed {len(removed_df)} rows.")


def clean_text(text):
    if pd.isna(text):
        return ""

    text = normalize_slang(text)
    tokens = text.lower().split()
    kept_tokens = []

    for token in tokens:
        raw_token = token

        if raw_token in english_words:
            kept_tokens.append(raw_token)
            continue

        cleaned = re.sub(r"^[^\w']+|[^\w']+$", '', raw_token)

        if cleaned in english_words or re.search(r"[\d@.:/+()'\-]", cleaned):
            kept_tokens.append(cleaned)

    return ' '.join(kept_tokens)

filtered_df = pd.read_csv('data.csv')

filtered_df['Semantically Cleaned Text'] = filtered_df['Extracted Text'].astype(str).apply(clean_text)

filtered_df.to_csv('semantically_cleaned_messages.csv', index=False)
