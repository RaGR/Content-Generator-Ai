import nltk
from nltk.corpus import wordnet
from transformers import T5Tokenizer, T5ForConditionalGeneration
import spacy
import pandas as pd
from csv import writer

# Ensure Needed NLTK Resources
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load Spacy English Model
spacy.cli.download("en_core_web_sm")
nlp = spacy.load('en_core_web_sm')

# Initialize T5 Model and Tokenizer for Text Generation
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

def generate_main_title(product_name):
    """Generate SEO-friendly main title"""
    return f"Buy {product_name} Online - Best {product_name} at Affordable Prices"

def generate_seo_keywords(product_name):
    """Simplified SEO keyword generation"""
    keywords = [product_name]
    synonyms = set()
    for syn in wordnet.synsets(product_name):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    keywords.extend(list(synonyms))
    return keywords[:10]  # Return top 10 keywords

def generate_seo_article(product_name, keywords):
    """Generate a basic SEO article using T5 for some flair"""
    input_text = f"Generate a product description about {product_name}, including {', '.join(keywords)}."
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=200, min_length=100, num_beams=4, no_repeat_ngram_size=2, early_stopping=True)
    article = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Basic structure addition
    article = f"# {product_name} Overview\n{article}\n\n## Features and Benefits\nPlease refer to the specifications below for more details."
    return article

def generate_summarized_article(article):
    """Summarize the article (Simplified)"""
    doc = nlp(article)
    summary = " ".join([sent.text for sent in doc.sents][:3])
    return summary

def generate_product_specifications(product_name):
    """Template-based product specifications (Simplified)"""
    specs = {
        "Product": product_name,
        "Weight": "Approx. 1 kg",
        "Dimensions": "10 x 10 x 20 cm",
        "Color": "Available in Black, White",
        "Material": "High-Quality Plastic"
    }
    return specs

def generate_seo_meta_description(product_name, keywords):
    """Generate SEO meta description"""
    return f"Discover the best {product_name} with {', '.join(keywords[:3])}. Buy now and experience the difference!"

def save_to_csv(data, filename):
    """Save data to a CSV file"""
    df = pd.DataFrame(data, index=[0])
    df.to_csv(filename, index=False)


def main():
    product_name = input("Enter the Product Name: ")
    
    main_title = generate_main_title(product_name)
    seo_keywords = generate_seo_keywords(product_name)
    seo_article = generate_seo_article(product_name, seo_keywords)
    summarized_article = generate_summarized_article(seo_article)
    product_specs = generate_product_specifications(product_name)
    meta_description = generate_seo_meta_description(product_name, seo_keywords)
    
    # Organize data for CSV
    data = {
        "Main Title": [main_title],
        "SEO Keywords": [", ".join(seo_keywords)],
        "SEO Article": [seo_article],
        "Summarized Article": [summarized_article],
        "Product Specifications": [str(product_specs)],
        "Meta Description": [meta_description]
    }
    
    save_to_csv(data, "generated_content.csv")
    print("Content Generated and Saved to generated_content.csv")

if __name__ == "__main__":
    main()
