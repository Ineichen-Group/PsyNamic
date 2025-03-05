import requests
import time
from lxml import etree as ET
import pandas as pd
import os

PUBMED_API_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
PUBMED_ABSTRACTS_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
SEARCH_STRING = '((Randomized Controlled Trial[Publication Type] OR Controlled Clinical Trial[Publication Type] OR Pragmatic Clinical Trial[Publication Type] OR Clinical Study[Publication Type] OR Adaptive Clinical Trial[Publication Type] OR Equivalence Trial[Publication Type] OR Clinical Trial[Publication Type] OR Clinical Trial, Phase I[Publication Type] OR Clinical Trial, Phase II[Publication Type] OR Clinical Trial, Phase III[Publication Type] OR Clinical Trial, Phase IV[Publication Type] OR Clinical Trial Protocol[Publication Type] OR multicenter study[Publication Type] OR "Clinical Studies as Topic"[Mesh] OR "Clinical Trials as Topic"[Mesh] OR "Clinical Trial Protocols as Topic"[Mesh] OR "Multicenter Studies as Topic"[Mesh] OR "Random Allocation"[Mesh] OR "Double-Blind Method"[Mesh] OR "Single-Blind Method"[Mesh] OR "Placebos"[Mesh:NoExp] OR "Control Groups"[Mesh] OR "Cross-Over Studies"[Mesh] OR random*[Title/Abstract] OR sham[Title/Abstract] OR placebo*[Title/Abstract] OR ((singl*[Title/Abstract] OR doubl*[Title/Abstract]) AND (blind*[Title/Abstract] OR dumm*[Title/Abstract] OR mask*[Title/Abstract])) OR ((tripl*[Title/Abstract] OR trebl*[Title/Abstract]) AND (blind*[Title/Abstract] OR dumm*[Title/Abstract] OR mask*[Title/Abstract])) OR "control study"[tiab:~3] OR "control studies"[tiab:~3] OR "control group"[tiab:~3] OR "control groups"[tiab:~3] OR "healthy volunteers"[tiab:~3] OR "control trial"[tiab:~3] OR "control trials"[tiab:~3] OR "controlled study"[tiab:~3] OR "controlled trial"[tiab:~3] OR "controlled studies"[tiab:~3] OR "controlled trials"[tiab:~3] OR "clinical study"[tiab:~3] OR "clinical studies"[tiab:~3] OR "clinical trial"[tiab:~3] OR "clinical trials"[tiab:~3] OR Nonrandom*[Title/Abstract] OR non random*[Title/Abstract] OR non-random*[Title/Abstract] OR quasi-random*[Title/Abstract] OR quasirandom*[Title/Abstract] OR "phase study"[tiab:~3] OR "phase studies"[tiab:~3] OR "phase trial"[tiab:~3] OR "phase trials"[tiab:~3] OR "crossover study"[tiab:~3] OR "crossover studies"[tiab:~3] OR "crossover trial"[tiab:~3] OR "crossover trials"[tiab:~3] OR "cross-over study"[tiab:~3] OR "cross-over studies"[tiab:~3] OR "cross-over trial"[tiab:~3] OR "cross-over trials"[tiab:~3] OR ((multicent*[tiab] OR multi-cent*[tiab] OR open label[tiab] OR open-label[tiab] OR equivalence[tiab] OR superiority[tiab] OR non-inferiority[tiab] OR noninferiority[tiab] OR quasiexperimental[tiab] OR quasi-experimental[tiab]) AND (study[tiab] OR studies[tiab] OR trial*[tiab])) OR allocated[tiab] OR pragmatic study[tiab] OR pragmatic studies[tiab] OR pragmatic trial*[tiab] OR practical trial*[tiab]) AND ("Hallucinogens"[Majr] OR "Lysergic Acid Diethylamide"[Majr] OR "Psilocybin"[Majr] OR "psilocin" [Supplementary Concept] OR "Mescaline"[Majr] OR "N,N-Dimethyltryptamine"[Majr] OR "Banisteriopsis"[Majr] OR "N-Methyl-3,4-methylenedioxyamphetamine"[Majr] OR "3,4-Methylenedioxyamphetamine"[Majr] OR ("Ketamine"[Majr] AND ("Behavioral Symptoms"[MeSH] OR "Mental Disorders"[Mesh])) OR "Ibogaine"[Majr] OR "salvinorin a"[Supplementary Concept] OR ((hallucinogen*[tiab] OR psychedel*[tiab] OR psychomimet*[tiab] OR entheo*[tiab] OR entactogen*[tiab]) AND (agent*[tiab] OR drug*[tiab] OR compound*[tiab] OR substance*[tiab] OR therap*[tiab] OR psychotherap*[tiab] OR medic*[tiab])) OR (LSD[tiab] AND (psychedel*[tiab] OR hallucinogen*[tiab] OR entheo*[tiab] OR trip*[tiab] OR psychiat*[tiab])) OR LSD-25[tiab] OR "lysergic acid diethylamide"[tiab] OR delysid*[tiab] OR lysergide[tiab] OR lysergamide[tiab] OR Psilocybin*[tiab] OR Psilocibin*[tiab] OR comp360[tiab] OR Psilocin*[tiab] OR 4-HO-DMT[tiab] OR psilocyn*[tiab] OR mescalin*[tiab] OR 3,4,5-trimethoxyphenethylamine[tiab] OR TMPEA[tiab] OR Peyot*[tiab] OR (DMT[tiab] AND (psychedel*[tiab] OR hallucinogen*[tiab] OR entheo*[tiab] OR trip*[tiab] OR psychiat*[tiab])) OR N,N-Dimethyltryptamine[tiab] OR dimethyltryptamine*[tiab] OR "dimethyl tryptamine"[tiab] OR N,N-DMT[tiab] OR ayahuasca[tiab] OR banisteriopsis[tiab] OR 5-methoxy-N,N-dimethyltryptamine[tiab] OR methylbufotenin[tiab] OR 5-MeO-DMT[tiab] OR "5 methoxy dmt"[tiab] OR "5 methoxy n, n dimethyl tryptamine"[tiab] OR "5 methoxydimethyltryptamine"[tiab] OR "n, n dimethyl 5 methoxytryptamine"[tiab] OR Methylenedioxymethamphetamine[tiab] OR "3,4-Methylenedioxy methamphetamine"[tiab] OR "n methyl 3, 4 methylenedioxyamphetamine"[tiab] OR midomafetamine[tiab] OR MDMA[tiab] OR (ecstasy[tiab] AND drug*[tiab]) OR ((Ketamin*[tiab] OR esketamine[tiab]) AND (psychedel*[tiab] OR hallucinogen*[tiab] OR entheo*[tiab] OR trip*[tiab] OR psychiat*[tiab])) OR Ibogaine[tiab] OR iboga[tiab] OR salvinorin[tiab] OR "salvia divinorum"[tiab])) NOT (("Animals"[Mesh] OR "Animal Experimentation"[Mesh] OR "Models, Animal"[Mesh] OR "Vertebrates"[Mesh]) NOT ("Humans"[Mesh] OR "Human Experimentation"[Mesh]))'


def get_pubmed_data(query_string: str, retstart: int = 0, retmax: int = 2000):
    """
    Get pubmed ids for a given query string (which includes the query and a time filter)
    """

    params = {
        'db': 'pubmed',
        'retmode': 'xml',
        'retstart': retstart,
        'retmax': retmax,
    }
    data = {
        'term': query_string,
    }

    try:
        response = requests.post(
            PUBMED_API_URL, params=params, data=data, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while querying PubMed: {e}")
        return None


def get_pubmed_abstracts(pmids: list):
    """
    Get abstracts and other metadata for a list of pubmed ids
    """
    pmids_str = ','.join(map(str, pmids))
    data = {
        'db': 'pubmed',
        'rettype': 'abstract',
        'id': pmids_str,
        'retmode': 'xml',
    }

    try:
        response = requests.post(PUBMED_ABSTRACTS_URL, data=data, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while fetching abstracts: {e}")
        return None


def parse_abstracts(xml_data: str):
    """
    Parse the XML data and extract the relevant information
    """
    root = ET.fromstring(xml_data.encode('utf-8'))
    abstracts = []

    for article in root.xpath('//PubmedArticle'):
        abstract_dict = {}

        abstract_dict['keywords'] = [
            a.text for a in article.xpath('.//Keyword') if a.text]

        article_list = article.find('.//PubmedData/ArticleIdList')
        if article_list is not None:
            pubmed_id_element = article_list.find(
                './/ArticleId[@IdType="pubmed"]')
            abstract_dict['pubmed_id'] = pubmed_id_element.text if pubmed_id_element is not None else None
            abstract_dict['pubmed_url'] = f'https://pubmed.ncbi.nlm.nih.gov/{abstract_dict["pubmed_id"]}/' if abstract_dict['pubmed_id'] else None

            doi_element = article_list.find('.//ArticleId[@IdType="doi"]')
            abstract_dict['doi'] = doi_element.text if doi_element is not None else None
        else:
            abstract_dict['pubmed_id'] = None
            abstract_dict['pubmed_url'] = None
            abstract_dict['doi'] = None

        pub_date = article.find(
            './/History/PubMedPubDate[@PubStatus="pubmed"]/Year')
        abstract_dict['year'] = pub_date.text if pub_date is not None else None

        title_element = article.find('.//Article/ArticleTitle')
        if title_element is None:
            abstract_dict['title'] = None
        else:
            abstract_dict['title'] = "".join(title_element.itertext())

        abstract_dict['abstract'] = None
        abstract_texts = article.findall('.//Article/Abstract/AbstractText')
        if abstract_texts:
            if len(abstract_texts) > 1:
                abstract = ''
                for a in abstract_texts:
                    abstract += a.get("Label", "") + ': ' + a.text + ' '
                abstract = abstract.strip()
                abstract_dict['abstract'] = abstract
            else:
                abstract = article.find('.//Article/Abstract/AbstractText')
                abstract_dict['abstract'] = "".join(abstract.itertext())

        authors = []
        for author in article.findall('.//Article/AuthorList/Author'):
            initials = author.find('.//Initials')
            last_name = author.find('.//LastName')
            collective_name = author.find('.//CollectiveName')

            if initials is not None and last_name is not None:
                authors.append(f"{initials.text}. {last_name.text}")
            elif collective_name is not None:
                authors.append(collective_name.text)

        abstract_dict['authors'] = ', '.join(authors) if authors else None
        abstracts.append(abstract_dict)

    return abstracts


def main():
    # Get dir of this file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    date_file = os.path.join(dir_path, 'last_data_fetch.txt')

    # open last_data_fetch.txt
    with open(date_file, 'r', encoding='utf-8') as f:
        last_data_fetch = f.read()

    search_string_with_date = f'{SEARCH_STRING} AND (("{last_data_fetch}"[Date - Publication] : "3000"[Date - Publication]))'

    start = 0
    total_results = 0
    all_abstracts = []
    count = 0

    start_time = time.time()

    while True:
        xml_data = get_pubmed_data(search_string_with_date, retstart=start)
        if xml_data:
            root = ET.fromstring(xml_data.encode('utf-8'))
            pmids = [
                id_element.text for id_element in root.xpath('//IdList/Id')]
            count = int(root.find('Count').text)

            # Step 2: Fetch abstracts for these articles
            if pmids:
                abstract_data = get_pubmed_abstracts(pmids)
                if abstract_data:
                    parsed_abstracts = parse_abstracts(abstract_data)
                    all_abstracts.extend(parsed_abstracts)
                    total_results += len(parsed_abstracts)

                    # If we reached the max number of results, stop
                    if total_results >= count:
                        break

                    start += 2000
                    # If we reached the max number of results, stop
                    if start >= count:
                        break

                    time.sleep(1)  # Delay to respect the rate limit
                else:
                    break
            else:
                break
        else:
            break
    
    end_time = time.time()
    # duration in format hh:mm:ss
    duration = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
    df = pd.DataFrame(all_abstracts)
    today = time.strftime("%Y/%m/%d")
    outfile = os.path.join(dir_path, f'pubmed_results_{today.replace("/", "")}_{duration}.csv')
    df.to_csv(outfile, index=False, encoding='utf-8')

    with open(date_file, 'w', encoding='utf-8') as f:
        f.write(today)


if __name__ == "__main__":
    main()
