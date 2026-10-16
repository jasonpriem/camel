# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========

import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

from camel.logger import get_logger
from camel.toolkits import BaseToolkit, FunctionTool
from camel.utils import MCPServer

logger = get_logger(__name__)

WORK_FIELDS = (
    "id,doi,display_name,publication_year,type,authorships,primary_location,"
    "open_access,cited_by_count,abstract_inverted_index"
)


def _work_path(identifier: str) -> str:
    r"""Builds the :obj:`/works/{id}` path from a DOI, DOI URL, OpenAlex
    ID or URL, :obj:`pmid:` or :obj:`pmcid:` identifier.

    Args:
        identifier (str): The identifier given by the caller.

    Returns:
        str: The URL path for the work.
    """
    value = identifier.strip()
    lower = value.lower()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "doi:",
    ):
        if lower.startswith(prefix):
            return "/works/doi:" + quote(value[len(prefix) :], safe="")
    if value.startswith("10."):
        return "/works/doi:" + quote(value, safe="")
    if lower.startswith(("https://openalex.org/", "http://openalex.org/")):
        value = value.rstrip("/").rsplit("/", 1)[1]
    return "/works/" + quote(value, safe=":")


def _rebuild_abstract(
    inverted_index: Optional[Dict[str, List[int]]],
) -> str:
    r"""Turns OpenAlex's :obj:`abstract_inverted_index` (each word
    mapped to its positions) back into plain text.

    Args:
        inverted_index (Optional[Dict[str, List[int]]]): The index.

    Returns:
        str: The abstract, or an empty string if there is none.
    """
    if not inverted_index:
        return ""
    positions = {
        i: word for word, idxs in inverted_index.items() for i in idxs
    }
    return " ".join(positions[i] for i in sorted(positions))


@MCPServer()
class OpenAlexToolkit(BaseToolkit):
    r"""A toolkit for searching OpenAlex, a free and open index of scholarly
    works (papers, books, datasets, preprints) across all fields, and for
    looking up a work by DOI, OpenAlex ID, PMID or PMCID.

    No API key is needed. Requests without one share a small free daily
    budget per IP address; a free key from
    https://openalex.org/settings/api gives your own budget.

    Args:
        api_key (Optional[str]): OpenAlex API key. If not given, the
            :obj:`OPENALEX_API_KEY` environment variable is used when set.
            (default: :obj:`None`)
        max_abstract_chars (int): Abstracts longer than this are cut off.
            (default: :obj:`2000`)
        timeout (Optional[float]): The timeout for API requests in seconds.
            (default: :obj:`None`)
    """

    BASE_URL = "https://api.openalex.org"

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_abstract_chars: int = 2000,
        timeout: Optional[float] = None,
    ) -> None:
        r"""Initializes the OpenAlexToolkit."""
        super().__init__(timeout=timeout)
        self.api_key = api_key or os.environ.get("OPENALEX_API_KEY")
        self.max_abstract_chars = max_abstract_chars

    def _get(
        self, path: str, params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        r"""Sends a GET request to the OpenAlex API.

        Args:
            path (str): The path after the base URL, e.g. :obj:`/works`.
            params (Dict[str, Any]): Query parameters.

        Returns:
            Optional[Dict[str, Any]]: The JSON response, or :obj:`None` if
                the API returned 404.
        """
        # The key goes in a header, not the URL, so it never appears in logs.
        headers = (
            {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        )
        response = requests.get(
            f"{self.BASE_URL}{path}",
            params=params,
            headers=headers,
            timeout=self.timeout,
        )
        if response.status_code == 404:
            return None
        if response.status_code == 429:
            raise RuntimeError(
                "OpenAlex rate limit or daily budget reached. Set "
                "OPENALEX_API_KEY to a free key from "
                "https://openalex.org/settings/api for your own budget."
            )
        response.raise_for_status()
        return response.json()

    def _format_work(self, work: Dict[str, Any]) -> Dict[str, Any]:
        r"""Flattens an OpenAlex work record into the fields an agent needs.

        Args:
            work (Dict[str, Any]): The work record from the API.

        Returns:
            Dict[str, Any]: The flattened record.
        """
        authors = [
            (authorship.get("author") or {}).get("display_name")
            for authorship in work.get("authorships") or []
        ]
        authors = [a for a in authors if a]
        if len(authors) > 10:
            authors = authors[:10] + [f"and {len(authors) - 10} more"]
        source = (work.get("primary_location") or {}).get("source") or {}
        open_access = work.get("open_access") or {}
        abstract = _rebuild_abstract(work.get("abstract_inverted_index"))
        if len(abstract) > self.max_abstract_chars:
            abstract = abstract[: self.max_abstract_chars] + "..."
        return {
            "openalex_id": work.get("id"),
            "doi": work.get("doi"),
            "title": work.get("display_name"),
            "publication_year": work.get("publication_year"),
            "type": work.get("type"),
            "authors": authors,
            "source": source.get("display_name"),
            "cited_by_count": work.get("cited_by_count"),
            "is_open_access": open_access.get("is_oa"),
            "open_access_url": open_access.get("oa_url"),
            "abstract": abstract,
        }

    def openalex_search_works(
        self,
        query: str,
        max_results: int = 10,
        from_year: Optional[int] = None,
        to_year: Optional[int] = None,
        open_access_only: bool = False,
    ) -> List[Dict[str, Any]]:
        r"""Search OpenAlex for scholarly works (papers, books, datasets,
        preprints) in any field.

        Args:
            query (str): Keywords to match in titles, abstracts and full
                text.
            max_results (int): Number of works to return, 1 to 100.
                (default: :obj:`10`)
            from_year (Optional[int]): Only works published in or after this
                year. (default: :obj:`None`)
            to_year (Optional[int]): Only works published in or before this
                year. (default: :obj:`None`)
            open_access_only (bool): Only works that are free to read.
                (default: :obj:`False`)

        Returns:
            List[Dict[str, Any]]: Works with title, authors, year, source,
                DOI, citation count, open-access URL and abstract. On failure,
                a single item with an :obj:`error` message.
        """
        filters = []
        if from_year and to_year:
            filters.append(f"publication_year:{from_year}-{to_year}")
        elif from_year:
            filters.append(f"publication_year:>{from_year - 1}")
        elif to_year:
            filters.append(f"publication_year:<{to_year + 1}")
        if open_access_only:
            filters.append("open_access.is_oa:true")
        params: Dict[str, Any] = {
            "search": query,
            "per_page": max(1, min(max_results, 100)),
            "select": WORK_FIELDS,
        }
        if filters:
            params["filter"] = ",".join(filters)
        try:
            data = self._get("/works", params) or {}
        except (requests.RequestException, RuntimeError) as e:
            logger.error(f"OpenAlex search failed: {e!s}")
            return [{"error": str(e)}]
        return [self._format_work(w) for w in data.get("results", [])]

    def openalex_get_work(self, identifier: str) -> Dict[str, Any]:
        r"""Look up one scholarly work in OpenAlex.

        Args:
            identifier (str): A DOI (e.g. :obj:`10.1038/nature12373` or
                :obj:`https://doi.org/...`), an OpenAlex work ID (e.g.
                :obj:`W2741809807`), :obj:`pmid:<PubMed ID>` or
                :obj:`pmcid:<PMC ID>`.

        Returns:
            Dict[str, Any]: The work's title, authors, year, source, DOI,
                citation count, open-access URL and abstract, or an
                :obj:`error` message if no work matches or the request
                fails.
        """
        try:
            work = self._get(_work_path(identifier), {"select": WORK_FIELDS})
        except (requests.RequestException, RuntimeError) as e:
            logger.error(f"OpenAlex lookup failed: {e!s}")
            return {"error": str(e)}
        if work is None:
            return {"error": f"No work found in OpenAlex for {identifier!r}."}
        return self._format_work(work)

    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of tools provided by the OpenAlex toolkit.

        Returns:
            List[FunctionTool]: List of available tools.
        """
        return [
            FunctionTool(self.openalex_search_works),
            FunctionTool(self.openalex_get_work),
        ]
