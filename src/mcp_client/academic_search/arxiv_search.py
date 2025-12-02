"""arXiv search MCP server implementation."""
import arxiv
from typing import List, Dict, Any, Optional


class ArxivSearchMCP:
    """MCP Server for searching arXiv academic papers."""
    
    def __init__(self):
        """Initialize the arXiv search MCP."""
        self.client = arxiv.Client()
    
    def search(
        self,
        query: str,
        *,
        max_results: int,
        search_in: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search arXiv for academic papers.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return (REQUIRED)
            search_in: List of fields to search in ('title', 'abstract', 'category')
                      If None, searches in all fields
        
        Returns:
            List of paper dictionaries with title, authors, abstract, url, etc.
        """
        if search_in is None:
            # Default fields - can be made configurable later
            search_in = ['title', 'abstract', 'category']
        
        # Build arXiv query based on search fields
        query_parts = []
        
        if 'title' in search_in:
            query_parts.append(f'ti:{query}')
        if 'abstract' in search_in:
            query_parts.append(f'abs:{query}')
        if 'category' in search_in:
            query_parts.append(f'cat:{query}')
        
        # Join query parts with OR
        arxiv_query = ' OR '.join(query_parts) if query_parts else query
        
        # Create search object
        search = arxiv.Search(
            query=arxiv_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        # Execute search and format results
        results = []
        try:
            for paper in self.client.results(search):
                results.append({
                    'title': paper.title,
                    'authors': [author.name for author in paper.authors],
                    'abstract': paper.summary,
                    'published': paper.published.strftime('%Y-%m-%d'),
                    'updated': paper.updated.strftime('%Y-%m-%d'),
                    'categories': paper.categories,
                    'url': paper.entry_id,
                    'pdf_url': paper.pdf_url
                })
        except Exception as e:
            # Return empty results on error
            print(f"arXiv search error: {e}")
            return []
        
        return results
    
    def format_results_for_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results into a readable context string for the LLM.
        
        Args:
            results: List of paper dictionaries from search()
        
        Returns:
            Formatted string containing paper information
        """
        if not results:
            return "No academic papers found for this query."
        
        context_parts = ["=== Academic Research Papers from arXiv ===\n"]
        
        for i, paper in enumerate(results, 1):
            context_parts.append(f"\n[{i}] {paper['title']}")
            context_parts.append(f"Authors: {', '.join(paper['authors'])}")
            context_parts.append(f"Published: {paper['published']}")
            context_parts.append(f"Categories: {', '.join(paper['categories'])}")
            context_parts.append(f"URL: {paper['url']}")
            context_parts.append(f"\nAbstract:\n{paper['abstract']}")
            context_parts.append("\n" + "-" * 80)
        
        context_parts.append("\n=== End of Academic Research Papers ===\n")
        
        return "\n".join(context_parts)


# Singleton instance
arxiv_mcp = ArxivSearchMCP()
