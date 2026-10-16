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


from unittest.mock import MagicMock, patch

import pytest
import requests

from camel.toolkits import OpenAlexToolkit
from camel.toolkits.openalex_toolkit import _work_path

WORK = {
    "id": "https://openalex.org/W3177828909",
    "doi": "https://doi.org/10.1038/s41586-021-03819-2",
    "display_name": "Highly accurate protein structure prediction with "
    "AlphaFold",
    "publication_year": 2021,
    "type": "article",
    "authorships": [
        {"author": {"display_name": f"Author {i}"}} for i in range(12)
    ],
    "primary_location": {"source": {"display_name": "Nature"}},
    "open_access": {"is_oa": True, "oa_url": "https://example.org/a.pdf"},
    "cited_by_count": 30000,
    "abstract_inverted_index": {
        "Proteins": [0],
        "are": [1],
        "essential": [2],
        "to": [3],
        "life": [4],
    },
}


def mock_response(status_code=200, payload=None):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(
            f"HTTP {status_code}"
        )
    return response


@pytest.fixture
def toolkit(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    return OpenAlexToolkit()


def test_get_tools(toolkit):
    names = {tool.func.__name__ for tool in toolkit.get_tools()}
    assert names == {"openalex_search_works", "openalex_get_work"}


@patch("requests.get")
def test_search_sends_filters_and_formats(mock_get, toolkit):
    mock_get.return_value = mock_response(payload={"results": [WORK]})
    works = toolkit.openalex_search_works(
        "alphafold",
        max_results=500,
        from_year=2020,
        to_year=2022,
        open_access_only=True,
    )
    kwargs = mock_get.call_args.kwargs
    assert mock_get.call_args.args[0] == "https://api.openalex.org/works"
    assert kwargs["params"]["search"] == "alphafold"
    assert kwargs["params"]["per_page"] == 100
    assert (
        kwargs["params"]["filter"]
        == "publication_year:2020-2022,open_access.is_oa:true"
    )
    assert kwargs["headers"] == {}
    work = works[0]
    assert work["title"].startswith("Highly accurate")
    assert work["source"] == "Nature"
    assert work["abstract"] == "Proteins are essential to life"
    assert work["authors"][-1] == "and 2 more"


@patch("requests.get")
def test_single_year_bounds(mock_get, toolkit):
    mock_get.return_value = mock_response(payload={"results": []})
    toolkit.openalex_search_works("x", from_year=2020)
    assert mock_get.call_args.kwargs["params"]["filter"] == (
        "publication_year:>2019"
    )
    toolkit.openalex_search_works("x", to_year=2020)
    assert mock_get.call_args.kwargs["params"]["filter"] == (
        "publication_year:<2021"
    )


@patch("requests.get")
def test_api_key_goes_in_header(mock_get, monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "test-key")
    mock_get.return_value = mock_response(payload={"results": []})
    OpenAlexToolkit().openalex_search_works("x")
    kwargs = mock_get.call_args.kwargs
    assert kwargs["headers"] == {"Authorization": "Bearer test-key"}
    assert "test-key" not in str(kwargs["params"])


@patch("requests.get")
def test_rate_limit_is_reported(mock_get, toolkit):
    mock_get.return_value = mock_response(status_code=429)
    result = toolkit.openalex_search_works("x")
    assert "OPENALEX_API_KEY" in result[0]["error"]


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        ("W3177828909", "/works/W3177828909"),
        ("https://openalex.org/W3177828909", "/works/W3177828909"),
        (
            "10.1038/s41586-021-03819-2",
            "/works/doi:10.1038%2Fs41586-021-03819-2",
        ),
        (
            "https://doi.org/10.1016/0040-1951(95)00196-4",
            "/works/doi:10.1016%2F0040-1951%2895%2900196-4",
        ),
        ("pmid:34265844", "/works/pmid:34265844"),
    ],
)
def test_work_path(identifier, expected):
    assert _work_path(identifier) == expected


@patch("requests.get")
def test_get_work(mock_get, toolkit):
    mock_get.return_value = mock_response(payload=WORK)
    work = toolkit.openalex_get_work("10.1038/s41586-021-03819-2")
    assert mock_get.call_args.args[0].endswith(
        "/works/doi:10.1038%2Fs41586-021-03819-2"
    )
    assert work["openalex_id"] == WORK["id"]


@patch("requests.get")
def test_get_work_not_found(mock_get, toolkit):
    mock_get.return_value = mock_response(status_code=404)
    result = toolkit.openalex_get_work("10.9999/missing")
    assert result["error"].startswith("No work found")
