import requests
from lxml import etree as ET

SEARCH_STRING = '((Randomized Controlled Trial[Publication Type] OR Controlled Clinical Trial[Publication Type] OR Pragmatic Clinical Trial[Publication Type] OR Clinical Study[Publication Type] OR Adaptive Clinical Trial[Publication Type] OR Equivalence Trial[Publication Type] OR Clinical Trial[Publication Type] OR Clinical Trial, Phase I[Publication Type] OR Clinical Trial, Phase II[Publication Type] OR Clinical Trial, Phase III[Publication Type] OR Clinical Trial, Phase IV[Publication Type] OR Clinical Trial Protocol[Publication Type] OR multicenter study[Publication Type] OR "Clinical Studies as Topic"[Mesh] OR "Clinical Trials as Topic"[Mesh] OR "Clinical Trial Protocols as Topic"[Mesh] OR "Multicenter Studies as Topic"[Mesh] OR "Random Allocation"[Mesh] OR "Double-Blind Method"[Mesh] OR "Single-Blind Method"[Mesh] OR "Placebos"[Mesh:NoExp] OR "Control Groups"[Mesh] OR "Cross-Over Studies"[Mesh] OR random*[Title/Abstract] OR sham[Title/Abstract] OR placebo*[Title/Abstract] OR ((singl*[Title/Abstract] OR doubl*[Title/Abstract]) AND (blind*[Title/Abstract] OR dumm*[Title/Abstract] OR mask*[Title/Abstract])) OR ((tripl*[Title/Abstract] OR trebl*[Title/Abstract]) AND (blind*[Title/Abstract] OR dumm*[Title/Abstract] OR mask*[Title/Abstract])) OR "control study"[tiab:~3] OR "control studies"[tiab:~3] OR "control group"[tiab:~3] OR "control groups"[tiab:~3] OR "healthy volunteers"[tiab:~3] OR "control trial"[tiab:~3] OR "control trials"[tiab:~3] OR "controlled study"[tiab:~3] OR "controlled trial"[tiab:~3] OR "controlled studies"[tiab:~3] OR "controlled trials"[tiab:~3] OR "clinical study"[tiab:~3] OR "clinical studies"[tiab:~3] OR "clinical trial"[tiab:~3] OR "clinical trials"[tiab:~3] OR Nonrandom*[Title/Abstract] OR non random*[Title/Abstract] OR non-random*[Title/Abstract] OR quasi-random*[Title/Abstract] OR quasirandom*[Title/Abstract] OR "phase study"[tiab:~3] OR "phase studies"[tiab:~3] OR "phase trial"[tiab:~3] OR "phase trials"[tiab:~3] OR "crossover study"[tiab:~3] OR "crossover studies"[tiab:~3] OR "crossover trial"[tiab:~3] OR "crossover trials"[tiab:~3] OR "cross-over study"[tiab:~3] OR "cross-over studies"[tiab:~3] OR "cross-over trial"[tiab:~3] OR "cross-over trials"[tiab:~3] OR ((multicent*[tiab] OR multi-cent*[tiab] OR open label[tiab] OR open-label[tiab] OR equivalence[tiab] OR superiority[tiab] OR non-inferiority[tiab] OR noninferiority[tiab] OR quasiexperimental[tiab] OR quasi-experimental[tiab]) AND (study[tiab] OR studies[tiab] OR trial*[tiab])) OR allocated[tiab] OR pragmatic study[tiab] OR pragmatic studies[tiab] OR pragmatic trial*[tiab] OR practical trial*[tiab]) AND ("Hallucinogens"[Majr] OR "Lysergic Acid Diethylamide"[Majr] OR "Psilocybin"[Majr] OR "psilocin" [Supplementary Concept] OR "Mescaline"[Majr] OR "N,N-Dimethyltryptamine"[Majr] OR "Banisteriopsis"[Majr] OR "N-Methyl-3,4-methylenedioxyamphetamine"[Majr] OR "3,4-Methylenedioxyamphetamine"[Majr] OR ("Ketamine"[Majr] AND ("Behavioral Symptoms"[MeSH] OR "Mental Disorders"[Mesh])) OR "Ibogaine"[Majr] OR "salvinorin a"[Supplementary Concept] OR ((hallucinogen*[tiab] OR psychedel*[tiab] OR psychomimet*[tiab] OR entheo*[tiab] OR entactogen*[tiab]) AND (agent*[tiab] OR drug*[tiab] OR compound*[tiab] OR substance*[tiab] OR therap*[tiab] OR psychotherap*[tiab] OR medic*[tiab])) OR (LSD[tiab] AND (psychedel*[tiab] OR hallucinogen*[tiab] OR entheo*[tiab] OR trip*[tiab] OR psychiat*[tiab])) OR LSD-25[tiab] OR "lysergic acid diethylamide"[tiab] OR delysid*[tiab] OR lysergide[tiab] OR lysergamide[tiab] OR Psilocybin*[tiab] OR Psilocibin*[tiab] OR comp360[tiab] OR Psilocin*[tiab] OR 4-HO-DMT[tiab] OR psilocyn*[tiab] OR mescalin*[tiab] OR 3,4,5-trimethoxyphenethylamine[tiab] OR TMPEA[tiab] OR Peyot*[tiab] OR (DMT[tiab] AND (psychedel*[tiab] OR hallucinogen*[tiab] OR entheo*[tiab] OR trip*[tiab] OR psychiat*[tiab])) OR N,N-Dimethyltryptamine[tiab] OR dimethyltryptamine*[tiab] OR "dimethyl tryptamine"[tiab] OR N,N-DMT[tiab] OR ayahuasca[tiab] OR banisteriopsis[tiab] OR 5-methoxy-N,N-dimethyltryptamine[tiab] OR methylbufotenin[tiab] OR 5-MeO-DMT[tiab] OR "5 methoxy dmt"[tiab] OR "5 methoxy n, n dimethyl tryptamine"[tiab] OR "5 methoxydimethyltryptamine"[tiab] OR "n, n dimethyl 5 methoxytryptamine"[tiab] OR Methylenedioxymethamphetamine[tiab] OR "3,4-Methylenedioxy methamphetamine"[tiab] OR "n methyl 3, 4 methylenedioxyamphetamine"[tiab] OR midomafetamine[tiab] OR MDMA[tiab] OR (ecstasy[tiab] AND drug*[tiab]) OR ((Ketamin*[tiab] OR esketamine[tiab]) AND (psychedel*[tiab] OR hallucinogen*[tiab] OR entheo*[tiab] OR trip*[tiab] OR psychiat*[tiab])) OR Ibogaine[tiab] OR iboga[tiab] OR salvinorin[tiab] OR "salvia divinorum"[tiab])) NOT (("Animals"[Mesh] OR "Animal Experimentation"[Mesh] OR "Models, Animal"[Mesh] OR "Vertebrates"[Mesh]) NOT ("Humans"[Mesh] OR "Human Experimentation"[Mesh]))'

last_data_fetch = "2024/01/01"
SEARCH_STRING += f'AND (("{last_data_fetch}"[Date - Publication] : "3000"[Date - Publication]))'
NR_RESULTS = 1000

def query_pubmed_api(query_string: str) -> list[int]:
    pubmed_api_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
    params = {
        'db': 'pubmed',
        'retmode': 'json',  # Changed 'format' to 'retmode' for consistency with the API documentation
        'retmax': 1000,
    }

    data = {
        'term': SEARCH_STRING,
    }

    try:
        response = requests.post(pubmed_api_url, params=params, data=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        nr_results = int(data['esearchresult']['count'])
        pmids = data['esearchresult']['idlist']
        if nr_results > NR_RESULTS:
            print(f"Found more than {NR_RESULTS} results: {nr_results}")
        return pmids
    except Exception as e:
        # Handle any exceptions (e.g., network errors, JSON parsing errors)
        print(f"Error occurred: {e}")

def query_pubmed_abstracts(pmids: list[int]):
    pubmed_api_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
    pmids_str = ','.join(map(str, pmids))
    data = {
        'db': 'pubmed',
        'rettype': 'abstract',
        'id': pmids_str
    }
    try:
        response = requests.post(pubmed_api_url, data=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        xml = response.text
        # Write xml in a file
        with open('abstracts.xml', 'w') as f:
            f.write(xml)
        root = ET.fromstring(xml)
        return root

    except Exception as e:
        print(f"Error occurred: {e}")
        return None
    
def process_abstract_xml(root: ET.Element):
    abstracts = []
    for article in root.xpath('//PubmedArticle'):
        breakpoint()
        abstract_dict = {}
        abstract_dict['pmid'] = int(article.find('.//PMID').text)
        abstract_dict['doi'] = article.find('.//ELocationID[@EIdType="doi"]').text
        abstract_dict['title'] = article.find('.//ArticleTitle').text
        abstract_dict['abstract'] = article.find('.//Abstract').text
        abstract = ''
        for abs in article.findall('.//AbstractText'):
            label = abs.get('Label')
            abstract += f'{label}: ' if label else ''
            abstract += abs.text
            abstract += ' '
        abstract.rstrip()

        # abstract_dict['abstract'] = article.find('Abstract/AbstractText').text
        # abstract_dict['authors'] = article.find('AuthorList').text
        # abstract_dict['keywords'] = article.find('KeywordList').text
        # abstract_dict['year'] = article.find('PubDate/Year').text
        # abstracts.append(abstract_dict)
    


    # abstract_dict = {
    #     'pmid': '',
    #     'doi': '',
    #     'title': '',
    #     'abstract': '',
    #     'authors': '',
    #     'keywords': '',
    #     'year': '',
    #         }

def main():
    pmids = query_pubmed_api(SEARCH_STRING)
    abstract_xml = query_pubmed_abstracts(pmids)
    process_abstract_xml(abstract_xml)
    print(abstracts)

if __name__ == '__main__':
    main()