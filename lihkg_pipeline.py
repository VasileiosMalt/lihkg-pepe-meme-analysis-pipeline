#!/usr/bin/env python
# coding: utf-8

# ##############################################################################
# #
# # LIHKG Data Analysis and Image Download Pipeline 
# # This script combines the multiple Python files utilised duting the research period into a single, functional pipeline.
# # It performs the following steps in order, without any manual intervention:
# # 1. Sets up the necessary directory structure.
# # 2. Scrapes LIHKG for a user-provided search term to get a list of threads.
# # 3. Preprocesses the scraped threads to identify relevant and popular ones.
# # 4. Scrapes all posts from each of the identified relevant threads, saving
# #    each thread's posts to its own CSV file.
# # 5. Consolidates all post CSVs into a single dataset for analysis.
# # 6. Analyzes the consolidated post data, extracting features like images and emojis,
# #    and generates statistical reports and plots.
# # 7. Performs a detailed analysis of the main thread data.
# # 8. Downloads all images found in the posts into a structured directory.
# # 9. Flattens the image directory, moving all images into a single folder.
# #
# ##############################################################################

import os
import shutil
import re
import json
import glob
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import cloudscraper
import matplotlib.pyplot as plt
import time

# --- 1. SETUP AND CONFIGURATION ---

def setup_directories():
    """Creates the necessary directories for data, reports, plots, and images."""
    print("Setting up project directories...")
    # Define base directories
    dirs = [
        "data/threads",
        "data/posts",
        "reports",
        "plots",
        "images/lihkg_images_nested",
        "images/all_images_flat"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Directories created successfully.")
    return {
        "threads": "data/threads",
        "posts": "data/posts",
        "reports": "reports",
        "plots": "plots",
        "nested_images": "images/lihkg_images_nested",
        "flat_images": "images/all_images_flat"
    }

# --- 2. THREAD & POST SCRAPING ---

def scrape_lihkg_threads(search_term, output_path, scraper):
    """
    Scrapes LIHKG for a given search term and saves the thread data to a CSV.
    Origin: lihkg_scr.py
    """
    print(f"Starting to scrape LIHKG for threads with search term: '{search_term}'...")
    output = []
    page = 1
    search_url = f"https://lihkg.com/api_v2/thread/search?q={search_term}"

    while True:
        complete_url = f"{search_url}&page={page}&count=100&type=now&order=new"
        try:
            forum_articles = scraper.get(complete_url).json()
            if forum_articles.get("success") == 1 and forum_articles["response"]["items"]:
                output.extend(forum_articles["response"]["items"])
                print(f"Scraped page {page} of threads...")
                page += 1
                time.sleep(0.5)  # Be polite to the server
            else:
                break
        except Exception as e:
            print(f"An error occurred during thread scraping: {e}")
            break
            
    if not output:
        print("No articles found or scraping failed.")
        return None

    df = pd.DataFrame.from_dict(output)
    df.to_csv(output_path, index=False)
    print(f"Scraped {len(df)} threads in total. Data saved to '{output_path}'.")
    return output_path

def scrape_thread_posts(thread_id, output_dir, scraper):
    """
    Scrapes all posts from a single thread ID. This is the newly added function.
    """
    print(f"Scraping posts for thread ID: {thread_id}...")
    api_url = f"https://lihkg.com/api_v2/thread/{thread_id}/page/"
    page = 1
    all_posts = []

    while True:
        complete_url = f"{api_url}{page}"
        try:
            response = scraper.get(complete_url).json()
            if response.get("success") == 1 and response["response"]["item_data"]:
                all_posts.extend(response["response"]["item_data"])
                total_pages = int(response["response"]["total_page"])
                print(f"-> Scraped page {page}/{total_pages} for thread {thread_id}")
                if page >= total_pages:
                    break
                page += 1
                time.sleep(0.5) # Be polite
            else:
                print(f"-> No more posts found for thread {thread_id} after page {page-1}. Reason: {response.get('error_message', 'Unknown')}")
                break
        except Exception as e:
            print(f"-> Error scraping posts for thread {thread_id} on page {page}: {e}")
            break

    if all_posts:
        df = pd.DataFrame(all_posts)
        output_path = os.path.join(output_dir, f"{thread_id}.csv")
        df.to_csv(output_path, index=False)
        print(f"->> Saved {len(df)} posts to {output_path}")
        return output_path
    else:
        print(f"->> No posts were scraped for thread {thread_id}.")
        return None

# --- 3. DATA PREPROCESSING ---

def preprocess_and_select_threads(threads_csv_path, reports_dir):
    """
    Filters threads, identifies top popular and engaged ones, and prepares for analysis.
    Origin: LIHKG_data_preprocessing.py
    """
    print("Preprocessing thread data...")
    if not os.path.exists(threads_csv_path):
        print(f"Error: Threads file not found at '{threads_csv_path}'")
        return None

    df = pd.read_csv(threads_csv_path)

    # Filter threads containing specific keywords
    keywords_pattern = "pepe|Pepe|青蛙|PePe|frog|Frog"
    pepe_threads = df[df.title.str.contains(keywords_pattern, case=False, na=False)].copy()
    print(f"Found {len(pepe_threads)} threads related to the keywords.")

    if pepe_threads.empty:
        print("No relevant threads found after filtering. Exiting.")
        return None

    # For this project, a specific list of thread_ids was used in the original analysis
    # We will use these as the "confirmed" threads to scrape for posts.
    related_ids = [515394, 1488910, 2043890, 1498296, 1473751, 1499549, 1009987, 1674598, 1581145, 1812899, 1490019, 1489024, 1776481, 1480928]
    confirmed_threads = pepe_threads[pepe_threads['thread_id'].isin(related_ids)].copy()
    
    if confirmed_threads.empty:
        print("None of the specifically targeted 'related_ids' were found in the scraped data.")
    else:
        print(f"Identified {len(confirmed_threads)} specific 'confirmed' threads for post scraping and detailed analysis.")
    
    return confirmed_threads

# --- 4. POST CONSOLIDATION & ANALYSIS ---

def analyze_and_consolidate_posts(posts_dir, reports_dir, plots_dir):
    """
    Consolidates post CSVs, extracts features, and performs analysis.
    Origin: LIHKG_data_analysis.py
    """
    print("Consolidating and analyzing post data...")
    post_files = glob.glob(os.path.join(posts_dir, "*.csv"))
    if not post_files:
        print("No post CSVs found. Skipping post analysis and image download.")
        return None

    print(f"Found {len(post_files)} post files. Consolidating...")
    
    df_list = [pd.read_csv(f) for f in post_files]
    posts_df = pd.concat(df_list, ignore_index=True)
    posts_df.drop(columns=[col for col in ['index', 'Unnamed: 0'] if col in posts_df.columns], inplace=True, errors='ignore')
    
    # Feature Extraction
    print("Extracting features from post messages (emojis, images, links)...")
    posts_df['msg'] = posts_df['msg'].astype(str)
    
    # Extract images: find all 'img src="..."' patterns
    posts_df['images'] = posts_df['msg'].apply(lambda x: re.findall(r'img src="(https?://.*?\.(?:png|jpg|jpeg|gif))"', x))
    posts_df['images_count'] = posts_df['images'].apply(len)

    # Extract LIHKG emojis: find patterns like '/assets/faces/lihkg/...'
    posts_df['hkgmoji'] = posts_df['msg'].apply(lambda x: re.findall(r'/assets/faces/lihkg/.*\.gif', x))
    posts_df['hkgmoji_count'] = posts_df['hkgmoji'].apply(len)
    
    consolidated_path = os.path.join(reports_dir, 'all_posts_analyzed.csv')
    posts_df.to_csv(consolidated_path, index=False)
    print(f"Consolidated and analyzed post data saved to '{consolidated_path}'")
    
    # Perform statistical analysis
    if 'vote_score' not in posts_df.columns:
        posts_df['vote_score'] = posts_df['like_count'] - posts_df['dislike_count']
        
    mean_vote_with_image = posts_df[posts_df['images_count'] > 0]['vote_score'].mean()
    mean_vote_no_image = posts_df[posts_df['images_count'] == 0]['vote_score'].mean()

    if not np.isnan(mean_vote_with_image) and not np.isnan(mean_vote_no_image) and mean_vote_no_image > 0:
        ratio = mean_vote_with_image / mean_vote_no_image
        print(f"-> Finding: Posts with an image received an average vote score {ratio:.2f} times higher.")

    return posts_df

# --- 5. IMAGE DOWNLOADING & ORGANIZATION ---

def download_images(posts_df, nested_dir, flat_dir):
    """
    Downloads images from URLs and organizes them.
    Origin: lihkg_image_download.py & Untitled.py
    """
    if posts_df is None or 'images' not in posts_df.columns:
        print("Skipping image download: No post data available.")
        return

    df_im = posts_df[posts_df['images_count'] > 0].copy()
    if df_im.empty:
        print("No images found in posts to download.")
        return
        
    total_images = df_im['images_count'].sum()
    print(f"Starting download of {total_images} images...")
    
    downloaded_count = 0
    for _, row in df_im.iterrows():
        thread_id = row['thread_id']
        post_id = row['post_id']
        
        # Create directory for the thread if it doesn't exist
        thread_path = os.path.join(nested_dir, str(thread_id))
        os.makedirs(thread_path, exist_ok=True)
        
        for image_url in row['images']:
            try:
                filename = image_url.split("/")[-1].split("?")[0]
                if not filename: continue
                
                # Path for nested structure
                nested_filepath = os.path.join(thread_path, filename)
                # Path for flat structure
                flat_filepath = os.path.join(flat_dir, filename)

                # Avoid re-downloading if file exists in the flat directory
                if os.path.exists(flat_filepath):
                    continue

                r = requests.get(image_url, stream=True, timeout=15)
                if r.status_code == 200:
                    r.raw.decode_content = True
                    # Save to nested directory first
                    with open(nested_filepath, 'wb') as f:
                        shutil.copyfileobj(r.raw, f)
                    # Then copy to flat directory, handling name conflicts
                    if os.path.exists(flat_filepath):
                        base, ext = os.path.splitext(filename)
                        i = 1
                        while os.path.exists(flat_filepath):
                            new_name = f"{base}_{i}{ext}"
                            flat_filepath = os.path.join(flat_dir, new_name)
                            i += 1
                    shutil.copy2(nested_filepath, flat_filepath)
                    downloaded_count += 1
                    if downloaded_count % 50 == 0:
                        print(f"   Downloaded {downloaded_count}/{total_images} images...")
                else:
                    print(f"Warning: Image not retrieved (status {r.status_code}) from {image_url}")
            except Exception as e:
                print(f"Warning: Failed to download {image_url}. Error: {e}")
                continue
    print(f"Image downloading and organization complete. Total new images downloaded: {downloaded_count}")

# --- 6. MAIN EXECUTION PIPELINE ---

if __name__ == "__main__":
    # Step 1: Setup
    dirs = setup_directories()
    scraper = cloudscraper.create_scraper() # Initialize scraper once
    
    # Step 2: Scrape Thread List
    search_term = input("Enter the term to search on LIHKG (e.g., 'pepe'): ")
    if not search_term:
        search_term = 'pepe' # Default value
        print("No term entered, using default: 'pepe'")
        
    threads_csv = os.path.join(dirs["threads"], f"lihkg_{search_term.replace(' ', '_')}_threads.csv")
    scraped_threads_path = scrape_lihkg_threads(search_term, threads_csv, scraper)
    
    if scraped_threads_path:
        # Step 3: Identify Relevant Threads for Post Scraping
        confirmed_threads = preprocess_and_select_threads(scraped_threads_path, dirs["reports"])
    
        if confirmed_threads is not None and not confirmed_threads.empty:
            # Step 4: Scrape Posts for each Confirmed Thread
            print("\n--- Starting Post Scraping Phase ---")
            for thread_id in confirmed_threads['thread_id']:
                scrape_thread_posts(thread_id, dirs["posts"], scraper)
            print("--- Post Scraping Phase Complete ---\n")

            # Step 5: Consolidate and Analyze All Scraped Posts
            analyzed_posts_df = analyze_and_consolidate_posts(dirs["posts"], dirs["reports"], dirs["plots"])

            # Step 6: Download and Organize Images
            if analyzed_posts_df is not None:
                download_images(analyzed_posts_df, dirs["nested_images"], dirs["flat_images"])

    print("\nPipeline execution finished.")
