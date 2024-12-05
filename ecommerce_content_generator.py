import openai
import csv
import os
from typing import List, Dict

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_content(prompt: str, max_tokens: int = 150) -> str:
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

def generate_main_title(product_name: str) -> str:
    prompt = f"Generate an SEO-friendly and appealing main title for the product: {product_name}"
    return generate_content(prompt, max_tokens=50)

def generate_seo_keywords(product_name: str) -> List[str]:
    prompt = f"Generate a list of relevant SEO keywords for the product: {product_name}. Prioritize keywords with high relevance and search volume."
    keywords = generate_content(prompt, max_tokens=100)
    return [keyword.strip() for keyword in keywords.split(',')]

def generate_seo_article(product_name: str, keywords: List[str]) -> str:
    prompt = f"Write a comprehensive SEO article about the product: {product_name}. Include the following keywords naturally: {', '.join(keywords)}. Ensure the article has a logical flow, with sections that highlight the product's features, benefits, and unique selling points."
    return generate_content(prompt, max_tokens=500)

def generate_summary(article: str) -> str:
    prompt = f"Summarize the following article, highlighting key points in a format suitable for quick reads or product descriptions:\n\n{article}"
    return generate_content(prompt, max_tokens=150)

def generate_product_specifications(product_name: str) -> str:
    prompt = f"Draft detailed product specifications for {product_name}, including technical details, features, and benefits. Use a structured, bulleted format for clarity."
    return generate_content(prompt, max_tokens=200)

def generate_meta_description(product_name: str, keywords: List[str]) -> str:
    prompt = f"Create an SEO meta description for {product_name} that includes essential keywords, is engaging, and fits within 150-160 characters. Keywords: {', '.join(keywords[:3])}"
    return generate_content(prompt, max_tokens=50)

def save_to_csv(data: Dict[str, str], filename: str = "product_content.csv"):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)
    print(f"Content saved to {filename}")

def main():
    product_name = input("Enter the product name: ")
    
    main_title = generate_main_title(product_name)
    seo_keywords = generate_seo_keywords(product_name)
    seo_article = generate_seo_article(product_name, seo_keywords)
    summary = generate_summary(seo_article)
    product_specs = generate_product_specifications(product_name)
    meta_description = generate_meta_description(product_name, seo_keywords)
    
    content = {
        "Product Name": product_name,
        "Main Title": main_title,
        "SEO Keywords": ", ".join(seo_keywords),
        "SEO Article": seo_article,
        "Summary": summary,
        "Product Specifications": product_specs,
        "Meta Description": meta_description
    }
    
    save_to_csv(content)

if __name__ == "__main__":
    main()