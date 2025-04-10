import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agno.tools import Toolkit
from agno.utils.log import log_debug, logger

try:
    import arxiv
except ImportError:
    raise ImportError("`arxiv` not installed. Please install using `pip install arxiv`")

try:
    from pypdf import PdfReader
except ImportError:
    raise ImportError("`pypdf` not installed. Please install using `pip install pypdf`")


class ArxivTools(Toolkit):
    def __init__(self, search_arxiv: bool = True, read_arxiv_papers: bool = True, download_dir: Optional[Path] = None):
        super().__init__(name="arxiv_tools")

        self.client: arxiv.Client = arxiv.Client()
        self.download_dir: Path = download_dir or Path(__file__).parent.joinpath("arxiv_pdfs")

        if search_arxiv:
            self.register(self.search_arxiv_and_return_articles)
        if read_arxiv_papers:
            self.register(self.read_arxiv_papers)

    def search_arxiv_and_return_articles(self, query: str, num_articles: int = 10) -> str:

        articles = []
        log_debug(f"Searching arxiv for: {query}")
        for result in self.client.results(
            search=arxiv.Search(
                query=query,
                max_results=num_articles,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending,
            )
        ):
            try:
                article = {
                    "title": result.title,
                    "id": result.get_short_id(),
                    "entry_id": result.entry_id,
                    "authors": [author.name for author in result.authors],
                    "primary_category": result.primary_category,
                    "categories": result.categories,
                    "published": result.published.isoformat() if result.published else None,
                    "pdf_url": result.pdf_url,
                    "links": [link.href for link in result.links],
                    "summary": result.summary,
                    "comment": result.comment,
                }
                articles.append(article)
            except Exception as e:
                logger.error(f"Error processing article: {e}")
        return json.dumps(articles, indent=4)

    def read_arxiv_papers(self, id_list: List[str], pages_to_read: Optional[int] = None) -> str:


        download_dir = self.download_dir
        download_dir.mkdir(parents=True, exist_ok=True)
        # articles_context = ("")

        articles = []
        log_debug(f"Searching arxiv for: {id_list}")
        for result in self.client.results(search=arxiv.Search(id_list=id_list)):
            try:
                article: Dict[str, Any] = {
                    "title": result.title,
                    "id": result.get_short_id(),
                    "entry_id": result.entry_id,
                    "authors": [author.name for author in result.authors],
                    "primary_category": result.primary_category,
                    "categories": result.categories,
                    "published": result.published.isoformat() if result.published else None,
                    "pdf_url": result.pdf_url,
                    "links": [link.href for link in result.links],
                    "summary": result.summary,
                    "comment": result.comment,
                }
                if result.pdf_url:
                    log_debug(f"Downloading: {result.pdf_url}")
                    pdf_path = result.download_pdf(dirpath=str(download_dir))
                    log_debug(f"To: {pdf_path}")
                    pdf_reader = PdfReader(pdf_path)
                    article["content"] = []
                    for page_number, page in enumerate(pdf_reader.pages, start=1):
                        if pages_to_read and page_number > pages_to_read:
                            break
                        content = {
                            "page": page_number,
                            "text": page.extract_text(),
                        }
                        article["content"].append(content)
                articles.append(article)
            except Exception as e:
                logger.error(f"Error processing article: {e}")
        return json.dumps(articles, indent=4)