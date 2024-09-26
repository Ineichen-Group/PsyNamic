from typing import Any
from django.core.management.base import BaseCommand, CommandParser
import pandas as pd
from psynamic.models import Study
from tqdm import tqdm
import requests


# python manage.py import_paper_details /home/vera/Documents/Arbeit/CRS/PsychNER/data/raw_data/asreview_dataset_all_Psychedelic\ Study.csv


class Command(BaseCommand):

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument('paper_csv', type=str)

    def handle(self, *args: Any, **options: Any) -> str | None:
        paper_df = pd.read_csv(options['paper_csv'])
        # iterate through all studies, use tqdm
        all_studies = Study.objects.all()
        for study in tqdm(all_studies, total=all_studies.count(), desc="Importing paper details"):
            try:
                row = paper_df[paper_df['record_id'] == study.id].iloc[0]
            except IndexError:
                print(f"Study with id {study.id} not found")
            # update study
            study.title = row['title']
            study.abstract = row['abstract']
            doi = row['doi']
            study.url = self.get_url(doi)
            study.doi = doi
            study.keywords = row['keywords']
            study.year = int(row['year'])
            study.save()

    @staticmethod
    def get_url(doi: str) -> str:
        """Get the link to pubmed of the article with the given DOI."""
        # PubMed API endpoint
        pubmed_api_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'

        # Parameters for the PubMed API request
        params = {
            'db': 'pubmed',
            'term': doi,
            'format': 'json'
        }

        try:
            # Send HTTP GET request to PubMed API
            response = requests.get(pubmed_api_url, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Parse JSON response
            data = response.json()

            if data['esearchresult']['idlist']:
                # Extract PubMed ID (PMID) from response
                pmid = data['esearchresult']['idlist'][0]

                # Construct PubMed URL
                pubmed_url = f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/'

                return pubmed_url
            else:
                return ''

        except Exception as e:
            # Handle any exceptions (e.g., network errors, JSON parsing errors)
            print(f"Error occurred: {e}")
            return ''
