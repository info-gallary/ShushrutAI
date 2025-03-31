import asyncio
from typing import Optional

from agno.tools import Toolkit

try:
    from crawl4ai import AsyncWebCrawler, CacheMode
except ImportError:
    raise ImportError("`crawl4ai` not installed. Please install using `pip install crawl4ai`")


class Crawl4aiTools(Toolkit):
    def __init__(
        self,
        max_length: Optional[int] = 1000,
    ):
        super().__init__(name="crawl4ai_tools")

        self.max_length = max_length

        self.register(self.web_crawler)

    def web_crawler(self, url: str, max_length: Optional[int] = None) -> str:

        if url is None:
            return "No URL provided"

        return asyncio.run(self._async_web_crawler(url, max_length))

    async def _async_web_crawler(self, url: str, max_length: Optional[int] = None) -> str:

        async with AsyncWebCrawler(thread_safe=True) as crawler:
            result = await crawler.arun(url=url, cache_mode=CacheMode.BYPASS)

            length = self.max_length or max_length
            if not result.markdown:
                return "No result"

            if length:
                result = result.markdown[:length]
                result = result.replace(" ", "")
                return result

            result = result.markdown.replace(" ", "")
        return result