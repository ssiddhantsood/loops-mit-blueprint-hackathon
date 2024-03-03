import os
import requests
from bs4 import BeautifulSoup
import serpapi
import openai
from dotenv import load_dotenv
from openai import OpenAI
import argparse
import json
from datetime import date
import re
from newspaper import Article
from dateutil.parser import parse
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import concurrent.futures
import numpy
from articleGrader import analyze_website_text
import threading
from multiprocessing import Process, Queue

# Load environment variables
load_dotenv()
serpapi_key = os.getenv('SERPAPI_KEY')
openai.api_key = os.getenv('OPENAI_KEY')
serpClient = serpapi.Client(api_key=serpapi_key)
nltk.download('punkt')  # For tokenizing words
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')


def load_user_preferences(json_filepath):
    with open(json_filepath, 'r') as file: user_preferences = json.load(file)
    return user_preferences


def scrape_search_results(query):
    result = serpClient.search(q=query, engine="google", location="Austin, Texas", hl="en", gl="us")
    urls = [r['link'] for r in result.get('organic_results', [])]
    return urls


def find_author(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Method 1: Meta tags (name=author)
        author = soup.find('meta', attrs={'name': 'author'})
        if author and author.has_attr('content'):
            return author['content']

        # Method 2: Schema.org JSON-LD structured data
        json_ld = soup.find('script', {'type': 'application/ld+json'})
        if json_ld:
            try:
                data = json.loads(json_ld.string)
                # Handle both single author and multiple authors
                if 'author' in data:
                    authors = data['author']
                    if isinstance(authors, list):
                        return ', '.join(author.get('name', '') for author in authors if author.get('name'))
                    elif 'name' in authors:
                        return authors['name']
            except json.JSONDecodeError:
                pass

        # Method 3: Common patterns in HTML
        for selector in ['article .author', '.article-author', 'footer .author', '.byline', '.author-name']:
            author_element = soup.select_one(selector)
            if author_element:
                author_name = author_element.get_text(strip=True)
                if author_name:
                    return author_name

        # Method 4: Heuristic-based search
        for text in soup.stripped_strings:
            if re.match("author[:]*", text.lower()):
                return re.sub("author[:]*", '', text, flags=re.IGNORECASE).strip()

    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")

    return "Author not found"


def find_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/',
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # This will raise an exception for 4xx and 5xx errors
        html_content = response.content
        soup = BeautifulSoup(html_content, 'html.parser')
        # Attempt to use Newspaper3k if available
        article = Article(url)
        article.set_html(html_content)
        article.parse()
        if article.text:
            return article.text
        # Fallback to BeautifulSoup if Newspaper3k fails
        content = None
        for selector in ['main', 'article', '[role="main"]', 'div.post', 'div.article']:
            content = soup.select_one(selector)
            if content:
                break
        if content:  # Ensure content is not None before calling get_text
            return ' '.join(p.text for p in content.find_all('p'))
        else:
            # As a last resort, use the text of the body, which may include non-content elements
            body = soup.body
            if body:
                return body.get_text(separator=' ', strip=True)
    except requests.RequestException as e:
        print(f"Failed to fetch article content for URL {url}: {e}")
    return ""


def find_date(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    # First attempt with Newspaper3k
    try:
        article = Article(url)
        article.download()
        article.parse()
        if article.publish_date:
            return article.publish_date
    except Exception as e:
        print(f"Error with Newspaper3k for URL {url}: {e}")

    # Fallback to BeautifulSoup
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Look for <meta> tags that might contain the publication date
        for tag in soup.find_all('meta'):
            if 'pubdate' in tag.get('property', '') or 'pubdate' in tag.get('name', '') or 'date' in tag.get('property',
                                                                                                             '') or 'date' in tag.get(
                    'name', ''):
                date_str = tag.get('content', '')
                try:
                    return parse(date_str)
                except ValueError:
                    pass

        # Look for text and tags that might contain the publication date
        date_selectors = ['time', '.published', '.pubdate', '.timestamp', '[itemprop="datePublished"]']
        for selector in date_selectors:
            date_element = soup.select_one(selector)
            if date_element:
                date_str = date_element.get('datetime') or date_element.get_text()
                try:
                    return parse(date_str)
                except ValueError:
                    pass

        # Regex search for dates within text nodes
        text_nodes = soup.find_all(string=re.compile(r'\b\d{4}-\d{1,2}-\d{1,2}\b'))
        for node in text_nodes:
            try:
                return parse(node)
            except ValueError:
                pass

    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")

    return "Date not found"


def stem_text(text):
    stemmer = PorterStemmer()
    word_tokens = word_tokenize(text)
    stemmed_text = ' '.join([stemmer.stem(word) for word in word_tokens])
    return stemmed_text


def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    lemmatized_text = ' '.join([lemmatizer.lemmatize(word) for word in word_tokens])
    return lemmatized_text


def clean_content(text):
    if text:
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text(separator=' ')
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\S+', '', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = text.lower()
        stop_words = set(stopwords.words('english'))
        word_tokens = text.split()
        filtered_text = ' '.join([word for word in word_tokens if word not in stop_words])
        # stemmed_text = stem_text(filtered_text)
        lemmatized_text = lemmatize_text(filtered_text)
        return lemmatized_text
    else:
        return ""


def extract_article_text(url, biasWeight):
    """
    Extract the main article text from a given URL.
    """
    articleInfo = {'author': '', 'content': '', 'date': '', 'cleanedInfo': ''}
    try:
        articleInfo['url'] = url
        articleInfo['title'] = get_website_title(url)
        articleInfo['content'] = find_content(url)
        articleInfo['date'] = find_date(url)
        articleInfo['cleanedInfo'] = clean_content(articleInfo['content'])
        articleInfo['breakdown'] = analyze_website_text(clean_text(articleInfo['content']), biasWeight)

    except Exception as e:
        print(f"Error extracting article text: {e}")
    return articleInfo


def extract_article_text_with_timeout(url, biasWeight, timeout=10):
    """
    Attempt to extract article text and stop if it takes more than 'timeout' seconds.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(extract_article_text, url, biasWeight)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"Extracting article text timed out after {timeout} seconds for URL: {url}")
            return ""


def generate_key_points(text):
    sentences = sent_tokenize(text)
    # Use BERT for keyword extraction
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    keyword_freq = X.sum(axis=1)
    # Get top N sentences based on keyword frequency
    top_sentences_idx = keyword_freq.argsort(axis=0)[-3:].flatten()
    top_sentences_idx = top_sentences_idx.tolist()[0]  # Convert to Python list
    key_points = [sentences[i] for i in top_sentences_idx]
    return key_points


def getAnswer(queue, query, key_points):
    flat_key_points = [item for sublist in key_points for item in sublist]
    key_points_text = "\n".join(flat_key_points)
    prompt = f"""
             Write an answer to the following: '{query}' using the following key points: {key_points_text}. \n

             Your answer should be logical. Furthermore, indicate two other "queries" that the would be helpful to search"""
    try:
        GPT4 = OpenAI(api_key=os.getenv('OPENAI_KEY'))
        completion = GPT4.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are an assistant who answers questions given some information scraped from web sources."},
                {"role": "user", "content": prompt}
            ]
        )
        key_points = completion.choices[0].message.content
        queue.put(('answer', key_points))
        # return key_points
    except Exception as e:
        print(f"Error generating answers: {e}")
        return ""

def clean_text(text):
    # Remove HTML tags
    cleaned_text = re.sub(r'<[^>]+>', '', text)
    # Remove newline characters
    cleaned_text = re.sub(r'\n', ' ', cleaned_text)
    return cleaned_text

def getMainAttribs(urls, biasWeight):
    all_key_points = []
    used_sources = []
    used_article_info = []
    successful_articles = 0  # Track the number of successfully processed articles

    for url in urls:
        # Break the loop if we've successfully processed 6 articles
        if successful_articles >= 6:
            break

        print(f"Processing URL: {url}")
        try:
            articleInfo = extract_article_text_with_timeout(url, biasWeight)
            if articleInfo and articleInfo['cleanedInfo']:
                key_points = generate_key_points(articleInfo['cleanedInfo'])
                if key_points:
                    all_key_points.append(key_points)
                    used_sources.append(url)
                    used_article_info.append(articleInfo)
                    successful_articles += 1  # Increment the counter for successful articles
        except Exception as e:
            print(f"Error processing URL {url}: {e}")


    return all_key_points, used_sources, reformatSourcesInfo(used_article_info)


def getPodcastMp3(queue, query, key_points):
    flat_key_points = [item for sublist in key_points for item in sublist]
    key_points_text = "\n".join(flat_key_points)
    prompt = f"""
             Create a podcast script based on the query: '{query}' and the following key points: {key_points_text}. \n
             The script should be engaging, easy to follow, and structured with an introduction, body, and conclusion. 
             Also, the script should be about 750 words. The final generated text should only include the words that the 
             host will say, and start every podcast with 'Welcome back to another Loops infocast'.
            """
    try:
        GPT4 = OpenAI(api_key=os.getenv('OPENAI_KEY'))
        completion = GPT4.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant who creates podcast scripts."},
                {"role": "user", "content": prompt}
            ]
        )
        script = completion.choices[0].message.content

        # fileName = f"{query}_infocast.mp3"
        # response = GPT4.audio.speech.create(model="tts-1", voice="alloy", input=script)
        # response.stream_to_file(fileName)

        queue.put(('mp3path', script))


    except Exception as e:
        print(f"Error generating podcast: {e}")
        return ""


def get_website_title(url):
    try:
        # Send a HTTP request to the URL
        response = requests.get(url, timeout=10)
        # Raise an exception if the request was unsuccessful
        response.raise_for_status()

        # Parse the content of the request with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the title tag
        title_tag = soup.find('title')

        # Return the text part of the title tag if it exists
        if title_tag:
            return title_tag.get_text()
        else:
            return "Title not found"
    except requests.RequestException as e:
        return f"Error fetching {url}: {e}"


# def getExtraResults(queue, urlList):
#     listOfBreakdowns = []
#     successful_articles = 0  # Track the number of successfully processed articles
#
#     for url in urlList:
#         if successful_articles >= 3:  # Break the loop if we've successfully processed 3 articles
#             break
#
#         try:
#             articleInfo = extract_article_text_with_timeout(url)
#             if articleInfo:  # Check if articleInfo is not None
#                 properBreakdown = {
#                     "date": articleInfo.get('date', ''),
#                     "title": articleInfo.get("title", ''),
#                     "url": articleInfo.get("url", ''),  # Assuming "info" was a typo and should be "url"
#                     'overall_bias_score': round(articleInfo.get('breakdown', {}).get('overall_bias_score', 0)),
#                     'overall_complexity_score': round(
#                         articleInfo.get('breakdown', {}).get('overall_complexity_score', 0)),
#                     'syntactic_complexity': articleInfo.get("syntactic_complexity", 0),
#                     'semantic_diversity': articleInfo.get("semantic_diversity", 0),
#                     'fk_grade': articleInfo.get("fk_grade", 0),
#                     'gunning_fog': articleInfo.get("gunning_fog", 0),
#                     'sortingHeuristic': articleInfo.get("sortingHeuristic", 0),
#                 }
#                 listOfBreakdowns.append(properBreakdown)
#                 successful_articles += 1  # Increment the counter for successful articles
#         except Exception as e:
#             print(f"Error processing URL {url}: {e}")
#             # Optionally, log the error or take other action. The loop will continue to the next URL.
#
#     queue.put(('extraResults', listOfBreakdowns))


def main(query, biasWeight):
    urls = scrape_search_results(query)
    all_key_points, used_sources, used_article_info = getMainAttribs(urls, biasWeight)
    queue = Queue()

    if all_key_points:
        processes = []
        answerProcess = Process(target=getAnswer, args=(queue, query, all_key_points))
        processes.append(answerProcess)
        answerProcess.start()

        podcastProcess = Process(target=getPodcastMp3, args=(queue, query, all_key_points))
        processes.append(podcastProcess)
        podcastProcess.start()

        # Collect results
        results = {}
        for _ in range(2):  # Number of processes
            key, value = queue.get()
            results[key] = value

        # Wait for all processes to complete-
        for process in processes:
            process.join()

        # Unpack results
        answerToQuery = results.get('answer')
        mp3path = results.get('mp3path')
        sorted_data = sorted(used_article_info, key=lambda x: x['sortingHeuristic'])

        return answerToQuery, sorted_data[:3], sorted_data[3:], mp3path

    else:
        print("No key points generated.")


def reformatSourcesInfo(sourcesInfo):
    listOfSourceDict = []
    for i in sourcesInfo:
        properBreakdown = {
            "date": i['date'],
            "title": i['title'],
            "url": i["url"],
            'overall_bias_score': round(i['breakdown']['overall_bias_score'], 1),
            'overall_readability_score': round(i['breakdown']['overall_readability_score'], 1),
            'syntactic_complexity': round(i['breakdown']['syntactic_complexity'], 1),
            'semantic_diversity': round(i['breakdown']['semantic_diversity'], 1),
            'fk_grade': round(i['breakdown']['fk_grade'], 1),
            'gunning_fog': round(i['breakdown']['gunning_fog'], 1),
            'sortingHeuristic': round(i['breakdown']['sortingHeuristic'], 1), }

        listOfSourceDict.append(properBreakdown)
    return listOfSourceDict


def getSearch(prompt, biasWeight=.6):
    answer, sourcesInfo, fullUrlList, mp3script = main(prompt, biasWeight)
    return answer, sourcesInfo, fullUrlList, mp3script

if __name__ == '__main__':
     answer, sourcesInfo, fullUrlList, mp3script  = getSearch("How do i become a better developer " )
     print(fullUrlList)