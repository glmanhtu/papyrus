# This script is for collecting the papyrus images from the University of Michigan Library
# The University of Michigan Library provides access to these materials for educational and research purposes.

import argparse
import json
import os
import urllib.parse

import PIL
import requests
import tqdm
from PIL import Image
from bs4 import BeautifulSoup
import re

from torch.utils.data import Dataset, DataLoader


PIL.Image.MAX_IMAGE_PIXELS = 933120000


def crawl_and_extract(url, pattern):
    # Make a GET request to the website
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad responses

    # Extract information using regular expression
    return re.findall(pattern, response.text)


def get_img_details(facet_quote, start=1, size=50):
    url = f"https://quod.lib.umich.edu/a/apis?fn1=apis_su;fq1={facet_quote};med=1;size={size};start={start};type=boolean;type=boolean;view=reslist;rgn1=ic_all;q1=*"
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad responses

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    items = soup.find_all('section', class_='results-list--small')
    result = []
    for item in items:
        result.append(item.a['href'])

    page_info = soup.find(id="results-pagination")
    if page_info is None:
        return result
    if start < int(page_info['max']):
        result += get_img_details(facet_quote, start + size, size)
    return result


def collect_im_links(im_links_dir):
    os.makedirs(im_links_dir, exist_ok=True)
    facets_file = os.path.join(im_links_dir, "all_facets.json")
    if os.path.exists(facets_file):
        with open(facets_file, 'r') as f:
            facets = json.load(f)
    else:
        facets_url = "https://quod.lib.umich.edu/cgi/i/image/image-idx?rgn1=ic_all&q1=*&med=1&type=boolean&c=apis&type=boolean&view=reslist&tpl=listallfacets&&focus=apis_su"

        # Regular expression pattern to extract information
        extraction_pattern = r'<input\stype="checkbox".*name="apis_su"\svalue="(.*)"\sdata-action'

        facets = crawl_and_extract(facets_url, extraction_pattern)
        with open(facets_file, 'w') as f:
            json.dump(facets, f)

    all_items = []
    for facet in tqdm.tqdm(facets):
        facet_quote = urllib.parse.quote_plus(facet)
        facet_json = os.path.join(im_links_dir, f"{facet_quote}.json")
        if not os.path.exists(facet_json):
            items = get_img_details(facet_quote)
            with open(facet_json, 'w') as f:
                json.dump(items, f)
        else:
            with open(facet_json, 'r') as f:
                items = json.load(f)

        all_items += items
    return all_items


def get_info(soup, key, dict_res):
    item = soup.find('div', attrs={"data-key": key})
    if item is not None:
        desc = item.find('dt').string
        value = item.find('dd').string
        dict_res[desc] = value


class IncorrectLinkException(Exception):
    ...


class DataCollector(Dataset):
    def __init__(self, im_links, im_folder):
        self.im_links = im_links
        self.im_folder = im_folder

    def __len__(self):
        return len(self.im_links)

    def get_img(self, im_link):
        response = requests.get(im_link)
        response.raise_for_status()  # Raise an exception for bad responses

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        download_button = soup.find(id='dropdown-action')
        if download_button is None:
            raise IncorrectLinkException(im_link)
        download_link = download_button.find('sl-menu').find_all('sl-menu-item')[-1]
        download_link = download_link['value']
        host_info = urllib.parse.urlparse(im_link)
        download_link = host_info.scheme + '://' + host_info.netloc + download_link
        try:
            im = Image.open(requests.get(download_link, stream=True).raw)
        except:
            raise IncorrectLinkException(im_link)

        im_info = {}
        get_info(soup, 'istruct_caption_apis_image_side', im_info)
        get_info(soup, 'apis_mat', im_info)
        get_info(soup, 'apis_da', im_info)
        get_info(soup, 'apis_pr', im_info)
        get_info(soup, 'apis_lang', im_info)
        get_info(soup, 'apis_g', im_info)
        get_info(soup, 'apis_au', im_info)
        get_info(soup, 'apis_ti', im_info)
        get_info(soup, 'apis_da1', im_info)
        get_info(soup, 'apis_da2', im_info)

        return im, im_info

    def __getitem__(self, idx):
        im_link = self.im_links[idx]
        im_name = os.path.splitext(im_link.split('/')[-1])[0]
        img_dir = os.path.join(self.im_folder, im_name)
        if os.path.exists(img_dir):
            return idx

        response = requests.get(im_link)
        response.raise_for_status()  # Raise an exception for bad responses

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        sections = soup.find_all('section', class_='records')

        groups = {}
        key, key_type = None, None
        for items in sections:
            for item in items:
                if item.name == 'h4':
                    pattern = r'^(\w+)\s\|\s(\w+)\s\('
                    res = re.search(pattern, item.string)
                    if res is not None:
                        key, key_type = res.group(1), res.group(2)
                if item.name == 'div' and key_type is not None:
                    for block in item.contents:
                        if block.name != 'div':
                            continue
                        link = block.a['href']
                        groups.setdefault(key, {}).setdefault(key_type, []).append(link)
                    key, key_type = None, None

        for group in groups:
            for group_type in groups[group]:
                for im_link in groups[group][group_type]:
                    try:
                        im, im_info = self.get_img(im_link)
                        im_name = os.path.splitext(im_link.split('/')[-1].split('?')[0])[0]
                        sample_dir = os.path.join(img_dir, group, group_type)
                        os.makedirs(sample_dir, exist_ok=True)
                        im.save(os.path.join(sample_dir, f'{im_name}.jpg'))
                        with open(os.path.join(sample_dir, f'{im_name}.json'), 'w') as f:
                            json.dump(im_info, f)
                    except IncorrectLinkException as e:
                        print(f'Unable to download image: {e}')

        return idx


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Michigan data collection script', add_help=False)
    parser.add_argument('--output-path', required=True, type=str, help='path to output dataset')
    args = parser.parse_args()

    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)
    im_links = collect_im_links(os.path.join(output_dir, 'im_links'))
    im_links = set([x.split('?')[0] for x in im_links])
    print(f'Total number of images: {len(im_links)}')

    im_dataset_path = os.path.join(output_dir, 'images')
    dataset = DataCollector(sorted(im_links), im_dataset_path)
    for idx in tqdm.tqdm(dataset):
        a = 1

