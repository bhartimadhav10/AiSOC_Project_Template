# gpu_rag_system.py
import os
import re
import glob
import time
import json
import requests
import networkx as nx
import concurrent.futures
import chromadb
import matplotlib.pyplot as plt
import hashlib
import logging
import fitz  # PyMuPDF
import torch
import numpy as np
import threading
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import warnings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from typing import List, Dict, Any, Optional

# Suppress warnings
warnings.filterwarnings("ignore")

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# Load environment variables
load_dotenv()

# --------------------------
# Configuration
# --------------------------
PDF_DIR = os.getenv("PDF_DIR", "uiet_pdfs")
CACHE_DIR = os.getenv("CACHE_DIR", "pdf_cache")
GRAPH_FILE = os.getenv("GRAPH_FILE", "knowledge_graph.json")
GRAPH_IMAGE = os.getenv("GRAPH_IMAGE", "knowledge_graph.png")
EXTRACTION_CACHE = os.getenv("EXTRACTION_CACHE", "extraction_cache.json")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", 50))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 10))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "phi3")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "llama3")
GRAPH_VISUALIZATION_LIMIT = int(os.getenv("GRAPH_VISUALIZATION_LIMIT", 200))
MAX_EXTRACTION_LENGTH = int(os.getenv("MAX_EXTRACTION_LENGTH", 6000))
EMBEDDING_CHUNK_SIZE = int(os.getenv("EMBEDDING_CHUNK_SIZE", 1500))
RETRY_LIMIT = int(os.getenv("RETRY_LIMIT", 3))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", 1.0))
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", 100))
API_PORT = int(os.getenv("API_PORT", 5000))
API_HOST = os.getenv("API_HOST", "0.0.0.0")
GPU_ENABLED = torch.cuda.is_available()

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("graph_rag_system.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"System Initialization - GPU Acceleration: {'ENABLED' if GPU_ENABLED else 'DISABLED'}")

# --------------------------
# Graph RAG System Class
# --------------------------
class GraphRAGSystem:
    def __init__(self):
        self.knowledge_graph = nx.DiGraph()
        self.documents: List[Dict] = []
        self.entity_index = defaultdict(list)
        self.embedding_model = None
        self.chroma_client = None
        self.chroma_collection = None
        self.extraction_cache = {}
        self.initialized = False
        self.device = self._get_device()
        
        # Create cache directories
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs(CHROMA_DIR, exist_ok=True)
        os.makedirs("static", exist_ok=True)

        # Load extraction cache if exists
        if os.path.exists(EXTRACTION_CACHE):
            try:
                with open(EXTRACTION_CACHE, "r", encoding="utf-8") as f:
                    self.extraction_cache = json.load(f)
                logger.info(f"Loaded extraction cache with {len(self.extraction_cache)} entries")
            except Exception as e:
                logger.error(f"Failed to load extraction cache: {str(e)}")
    
    def _get_device(self) -> torch.device:
        """Get the best available device (GPU or CPU)"""
        if GPU_ENABLED:
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        logger.info("Using CPU for processing")
        return torch.device("cpu")
    
    def save_extraction_cache(self) -> None:
        """Save extraction results to cache file"""
        try:
            with open(EXTRACTION_CACHE, "w", encoding="utf-8") as f:
                json.dump(self.extraction_cache, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved extraction cache with {len(self.extraction_cache)} entries")
        except Exception as e:
            logger.error(f"Failed to save extraction cache: {str(e)}")
    
    def initialize(self) -> bool:
        """Initialize the system with GPU acceleration"""
        if not self.initialized:
            logger.info("Initializing system components...")
            if self._initialize_embeddings():
                self._load_or_build_graph()
                self.initialized = True
                logger.info("System initialization completed successfully")
            else:
                logger.error("System initialization failed")
        return self.initialized
    
    def _initialize_embeddings(self) -> bool:
        """Initialize the embedding model and ChromaDB with GPU support"""
        try:
            # Load embedding model with GPU support
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=self.device)
            logger.info(f"Embedding model loaded on {self.device}")
            
            # Initialize ChromaDB
            logger.info("Initializing ChromaDB vector store...")
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
            
            # Create embedding function
            embedding_function = SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL,
                device="cuda" if GPU_ENABLED else "cpu"
            )
            
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="uiet_documents",
                embedding_function=embedding_function
            )
            logger.info("ChromaDB collection initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            return False

    def _load_or_build_graph(self) -> None:
        """Load existing graph or build a new one with GPU acceleration"""
        if self._load_graph():
            logger.info("Knowledge graph loaded from file")
            # Still need to build vector index if not done
            if self.chroma_collection.count() == 0:
                logger.info("Building vector index from existing documents...")
                if self._extract_text_from_pdfs_parallel():
                    self._build_vector_index()
        else:
            logger.info("Building new knowledge graph...")
            if self._extract_text_from_pdfs_parallel():
                if self._extract_entities_relations():
                    self._save_graph()
                    self._build_vector_index()
                    self._visualize_graph()
                    self.save_extraction_cache()
                    logger.info("Knowledge graph construction completed")
                else:
                    logger.error("Failed to extract entities and relations")
            else:
                logger.error("Failed to process PDFs")

    def _extract_text_from_pdf(self, pdf_path: str) -> Dict:
        """Fast text extraction using PyMuPDF"""
        filename = os.path.basename(pdf_path)
        cache_file = os.path.join(CACHE_DIR, f"{filename}.txt")
        
        # Use cached text if available
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    text = f.read()
                return {"id": filename, "text": text, "path": pdf_path}
            except Exception as e:
                logger.error(f"Error reading cache for {filename}: {str(e)}")
        
        logger.info(f"Processing: {filename}")
        text = ""
        
        try:
            # Use PyMuPDF with optimized settings
            with fitz.open(pdf_path) as doc:
                num_pages = min(doc.page_count, MAX_PDF_PAGES)
                for page_num in range(num_pages):
                    page = doc.load_page(page_num)
                    text += page.get_text("text", sort=True) + "\n\n"
            
            # Clean text
            text = re.sub(r'\s+', ' ', text).strip()
        except Exception as e:
            logger.error(f"PDF processing failed for {filename}: {str(e)}")
            return {"id": filename, "text": "", "path": pdf_path}
        
        # Save to cache
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            logger.error(f"Failed to cache text for {filename}: {str(e)}")
            
        return {"id": filename, "text": text, "path": pdf_path}
    
    def _extract_text_from_pdfs_parallel(self) -> bool:
        """Process PDFs in parallel for maximum speed"""
        logger.info(f"Scanning for PDFs in: {PDF_DIR}")
        pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
        
        if not pdf_files:
            logger.error("No PDF files found.")
            return False
        
        logger.info(f"Found {len(pdf_files)} PDF files for processing")
        
        # Use thread-based parallelism
        num_workers = min(os.cpu_count() * 2, 16)
        logger.info(f"Using {num_workers} parallel threads for PDF extraction")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self._extract_text_from_pdf, file): file for file in pdf_files}
            results = []
            
            for future in concurrent.futures.as_completed(futures):
                file = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {file}: {str(e)}")
                    results.append({"id": os.path.basename(file), "text": "", "path": file})
        
        # Filter out empty documents
        self.documents = [
            doc for doc in results 
            if doc and doc.get('text') and len(doc['text'].strip()) >= MIN_TEXT_LENGTH
        ]
        logger.info(f"Successfully processed {len(self.documents)} documents")
        return len(self.documents) > 0

    def _extract_entities_relations(self) -> bool:
        """Batch process knowledge extraction for maximum speed"""
        logger.info("Starting knowledge extraction process...")
        successful = 0
        total_docs = len(self.documents)
        
        if total_docs == 0:
            logger.error("No documents available for extraction")
            return False
        
        # Create document hashes for caching
        doc_hashes = {
            doc['id']: hashlib.md5(doc['text'].encode()).hexdigest()
            for doc in self.documents
        }
        
        # Process cached documents first
        for doc in self.documents:
            doc_hash = doc_hashes[doc['id']]
            if doc_hash in self.extraction_cache:
                self._add_to_graph(doc['id'], self.extraction_cache[doc_hash])
                successful += 1
        
        # Get remaining documents to process
        remaining_docs = [
            doc for doc in self.documents 
            if doc_hashes[doc['id']] not in self.extraction_cache
        ]
        
        logger.info(f"{len(remaining_docs)} documents require knowledge extraction")
        
        # Process in batches for efficiency
        for i in range(0, len(remaining_docs), BATCH_SIZE):
            batch_docs = remaining_docs[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (len(remaining_docs) // BATCH_SIZE) + 1
            logger.info(f"Processing extraction batch {batch_num}/{total_batches} ({len(batch_docs)} docs)")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, BATCH_SIZE)) as executor:
                futures = {
                    executor.submit(self._extract_from_document, doc['id'], doc['text'][:MAX_EXTRACTION_LENGTH]): doc
                    for doc in batch_docs
                }
                
                for future in concurrent.futures.as_completed(futures):
                    doc = futures[future]
                    try:
                        knowledge = future.result()
                        if knowledge:
                            doc_hash = doc_hashes[doc['id']]
                            self.extraction_cache[doc_hash] = knowledge
                            self._add_to_graph(doc['id'], knowledge)
                            successful += 1
                    except Exception as e:
                        logger.error(f"Extraction failed for {doc['id']}: {str(e)}")
            
            # Save cache after each batch
            self.save_extraction_cache()
            time.sleep(REQUEST_DELAY * 2)
        
        logger.info(f"Knowledge extracted from {successful}/{total_docs} documents")
        return successful > 0
    
    def _extract_from_document(self, doc_id: str, text: str) -> Optional[Dict]:
        """Extract knowledge from a document with retry logic"""
        if not text.strip():
            return None
            
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.extraction_cache:
            return self.extraction_cache[text_hash]
        
        # Optimized prompt for extraction
        prompt = (
            f"Analyze this academic text and extract key entities and relationships:\n\n"
            f"### INSTRUCTIONS:\n"
            f"1. Identify departments, courses, faculty, research areas\n"
            f"2. Extract relationships in [source, type, target] format\n"
            f"3. Return JSON: {{\"entities\": [\"...\"], \"relations\": [[\"...\", \"...\", \"...\"]]}}\n\n"
            f"TEXT:\n{text[:MAX_EXTRACTION_LENGTH]}\n\n"
            f"JSON:"
        )
        
        # Try with retries
        for attempt in range(1, RETRY_LIMIT + 1):
            try:
                response = requests.post(
                    OLLAMA_ENDPOINT,
                    json={
                        "model": EXTRACTION_MODEL,
                        "prompt": prompt,
                        "format": "json",
                        "stream": False,
                        "options": {
                            "num_ctx": 8192,
                            "temperature": 0.2,
                            "num_gpu": 50 if GPU_ENABLED else 0
                        }
                    },
                    timeout=180 + (30 * attempt)
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "response" in result:
                        knowledge = self._robust_json_parse(result["response"].strip())
                        if knowledge:
                            return knowledge
                else:
                    logger.warning(f"Ollama error for {doc_id}: HTTP {response.status_code} (attempt {attempt})")
            except requests.exceptions.Timeout:
                logger.warning(f"Ollama timeout for {doc_id} (attempt {attempt})")
            except Exception as e:
                logger.error(f"Request failed for {doc_id}: {str(e)} (attempt {attempt})")
            
            # Exponential backoff before retry
            time.sleep(5 * attempt)
        
        logger.error(f"All extraction attempts failed for {doc_id}")
        return self._fallback_extraction(text)

    def _robust_json_parse(self, response_text: str) -> Dict:
        """Robust JSON parsing with error correction"""
        try:
            return json.loads(response_text)
        except:
            # Fallback to simple extraction
            entities = re.findall(r'"entities"\s*:\s*\[(.*?)\]', response_text, re.DOTALL)
            relations = re.findall(r'"relations"\s*:\s*\[(.*?)\]', response_text, re.DOTALL)
            
            entity_list = []
            relation_list = []
            
            if entities:
                entity_list = [e.strip(' "') for e in entities[0].split(',') if e.strip()]
            
            if relations:
                for rel in relations[0].split('],['):
                    parts = [p.strip(' "') for p in rel.strip('[] ').split(',')]
                    if len(parts) >= 3:
                        relation_list.append(parts[:3])
            
            return {"entities": entity_list, "relations": relation_list}
    
    def _fallback_extraction(self, text: str) -> Dict:
        """Fast rule-based extraction"""
        entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        relations = []
        
        for match in re.findall(r'(\w+)\s+(teaches|lectures on|research area|works at|studies|offers|requires)\s+(\w+)', text, re.IGNORECASE):
            relations.append([match[0], match[1], match[2]])
        
        return {"entities": list(set(entities)), "relations": relations}
    
    def _add_to_graph(self, doc_id: str, knowledge: Dict) -> None:
        """Add extracted knowledge to the graph with validation checks"""
        # Add entities
        for entity in knowledge.get("entities", []):
            if not entity or len(entity) < 3:
                continue
                
            entity_clean = re.sub(r'[^\w\s]', '', entity).title()
            if entity_clean:
                if entity_clean not in self.knowledge_graph:
                    self.knowledge_graph.add_node(entity_clean, type="concept", documents=[doc_id])
                elif doc_id not in self.knowledge_graph.nodes[entity_clean].get("documents", []):
                    self.knowledge_graph.nodes[entity_clean]["documents"].append(doc_id)
        
        # Add relationships with validation checks
        for relation in knowledge.get("relations", []):
            if len(relation) < 3:
                continue
                
            source = re.sub(r'[^\w\s]', '', relation[0]).title() if relation[0] else None
            target = re.sub(r'[^\w\s]', '', relation[2]).title() if relation[2] else None
            rel_type = str(relation[1]).lower() if relation[1] is not None else "related_to"
            
            if not source or not target or not rel_type:
                continue
                
            if source not in self.knowledge_graph:
                self.knowledge_graph.add_node(source, type="concept", documents=[doc_id])
            if target not in self.knowledge_graph:
                self.knowledge_graph.add_node(target, type="concept", documents=[doc_id])
            
            if self.knowledge_graph.has_edge(source, target):
                self.knowledge_graph[source][target]["weight"] += 1
            else:
                self.knowledge_graph.add_edge(source, target, relationship=rel_type, weight=1, documents=[doc_id])

    def _build_vector_index(self) -> bool:
        """Build vector index with GPU acceleration"""
        if not self.documents:
            logger.error("No documents available for indexing")
            return False
            
        logger.info("Building vector index with GPU acceleration...")
        
        if self.chroma_collection.count() > 0:
            logger.info("Vector index already exists")
            return True
            
        chunks = []
        metadatas = []
        ids = []
        
        # Prepare documents for indexing
        for doc in self.documents:
            text = doc['text']
            
            # Split text into chunks
            start = 0
            while start < len(text):
                end = min(start + EMBEDDING_CHUNK_SIZE, len(text))
                chunk = text[start:end]
                
                # Try to end at sentence boundary
                if end < len(text):
                    last_period = chunk.rfind('.')
                    if last_period > EMBEDDING_CHUNK_SIZE * 0.8:
                        chunk = chunk[:last_period + 1]
                        end = start + len(chunk)
                
                chunks.append(chunk)
                metadatas.append({
                    "doc_id": doc['id'],
                    "source": "pdf",
                    "path": doc['path']
                })
                ids.append(f"{doc['id']}_{len(ids)}")
                
                start = end
                if start >= len(text):
                    break
        
        # Add to ChromaDB in batches
        batch_size = 500
        total_chunks = len(chunks)
        logger.info(f"Indexing {total_chunks} text chunks...")
        
        for i in range(0, total_chunks, batch_size):
            batch_end = min(i + batch_size, total_chunks)
            self.chroma_collection.add(
                documents=chunks[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
            logger.info(f"Indexed {batch_end}/{total_chunks} chunks")
        
        logger.info(f"Vector index built with {total_chunks} chunks")
        return True

    def _save_graph(self) -> bool:
        """Save the knowledge graph to a file"""
        graph_data = {
            "nodes": [{"id": n, **data} for n, data in self.knowledge_graph.nodes(data=True)],
            "edges": [{"source": u, "target": v, **data} for u, v, data in self.knowledge_graph.edges(data=True)]
        }
        
        try:
            with open(GRAPH_FILE, "w", encoding="utf-8") as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Knowledge graph saved to {GRAPH_FILE}")
            return True
        except Exception as e:
            logger.error(f"Failed to save graph: {str(e)}")
            return False
    
    def _load_graph(self) -> bool:
        """Load knowledge graph from file"""
        if os.path.exists(GRAPH_FILE):
            try:
                with open(GRAPH_FILE, "r", encoding="utf-8") as f:
                    graph_data = json.load(f)
                
                self.knowledge_graph = nx.DiGraph()
                
                # Add nodes
                for node in graph_data["nodes"]:
                    node_id = node.pop("id")
                    self.knowledge_graph.add_node(node_id, **node)
                
                # Add edges
                for edge in graph_data["edges"]:
                    source = edge.pop("source")
                    target = edge.pop("target")
                    self.knowledge_graph.add_edge(source, target, **edge)
                
                logger.info(f"Loaded knowledge graph with {len(self.knowledge_graph.nodes)} nodes and {len(self.knowledge_graph.edges)} edges")
                return True
            except Exception as e:
                logger.error(f"Error loading graph: {str(e)}")
        return False

    def _visualize_graph(self) -> bool:
        """Create a high-quality visualization of the knowledge graph"""
        if len(self.knowledge_graph.nodes) == 0:
            logger.error("No nodes to visualize!")
            return False
            
        logger.info("Generating graph visualization...")
        
        if len(self.knowledge_graph.nodes) > GRAPH_VISUALIZATION_LIMIT:
            logger.info(f"Graph too large ({len(self.knowledge_graph.nodes)} nodes). Visualizing top {GRAPH_VISUALIZATION_LIMIT} nodes.")
            degrees = dict(self.knowledge_graph.degree())
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:GRAPH_VISUALIZATION_LIMIT]
            G = self.knowledge_graph.subgraph(top_nodes)
        else:
            G = self.knowledge_graph
            
        plt.figure(figsize=(24, 18), dpi=300)
        pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)
        degrees = dict(G.degree())
        node_sizes = [np.sqrt(degrees[n] + 1) * 300 for n in G.nodes()]
        
        nx.draw_networkx_nodes(
            G, 
            pos, 
            node_size=node_sizes,
            node_color="skyblue",
            alpha=0.9,
            edgecolors="black",
            linewidths=0.5
        )
        
        edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [w / max_weight * 3 + 0.5 for w in edge_weights]
        
        nx.draw_networkx_edges(
            G, 
            pos, 
            edge_color="gray",
            width=edge_widths,
            alpha=0.7,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=12
        )
        
        edge_labels = nx.get_edge_attributes(G, 'relationship')
        nx.draw_networkx_edge_labels(
            G, 
            pos, 
            edge_labels=edge_labels,
            font_color='darkred',
            font_size=8,
            bbox=dict(alpha=0.8, facecolor="white", edgecolor="none", boxstyle="round,pad=0.2")
        )
        
        nx.draw_networkx_labels(
            G, 
            pos, 
            font_size=10,
            font_family="sans-serif",
            font_weight="bold",
            alpha=0.9
        )
        
        plt.title("UIET Knowledge Graph", fontsize=24, pad=20)
        plt.figtext(0.5, 0.01, 
                   f"Nodes: {len(G.nodes)} | Edges: {len(G.edges)} | "
                   f"Edge thickness = relationship frequency", 
                   ha="center", fontsize=10)
        
        plt.axis("off")
        plt.tight_layout()
        output_path = os.path.join("static", GRAPH_IMAGE)
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
        logger.info(f"Graph visualization saved to {output_path}")
        return True

    def get_graph_stats(self) -> Dict:
        """Get detailed statistics about the knowledge graph"""
        stats = {
            "nodes": len(self.knowledge_graph.nodes),
            "edges": len(self.knowledge_graph.edges),
            "top_entities": [],
            "relationships": []
        }
        
        if stats["nodes"] > 0:
            degree_centrality = nx.degree_centrality(self.knowledge_graph)
            top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            
            stats["top_entities"] = [
                {"entity": node, "centrality": float(centrality)}
                for node, centrality in top_nodes
            ]
            
            edges = list(self.knowledge_graph.edges(data=True))[:5]
            stats["relationships"] = [
                {"source": u, "target": v, "relationship": data['relationship'], "weight": data.get('weight', 1)}
                for u, v, data in edges
            ]
        
        return stats

    def query_with_rag(self, question: str) -> str:
        """Fast question answering with GPU acceleration"""
        start_time = time.time()
        logger.info(f"Processing query: {question}")
        
        # Step 1: Fast graph context retrieval
        graph_context = self._retrieve_graph_context(question)
        
        # Step 2: Semantic document retrieval
        relevant_docs = self._retrieve_relevant_documents_chroma(question)
        
        # Step 3: Generate answer with GPU acceleration
        answer = self._generate_structured_answer(question, graph_context, relevant_docs)
        
        logger.info(f"Response generated in {time.time() - start_time:.2f} seconds")
        return answer
    
    def _retrieve_graph_context(self, question: str) -> List[str]:
        """Fast context retrieval from knowledge graph"""
        question_lower = question.lower()
        context = set()
        
        for entity in self.knowledge_graph.nodes:
            if entity.lower() in question_lower:
                for neighbor in self.knowledge_graph.neighbors(entity):
                    if self.knowledge_graph.has_edge(entity, neighbor):
                        rel = self.knowledge_graph[entity][neighbor]["relationship"]
                        context.add(f"{entity} -> {rel} -> {neighbor}")
        
        return list(context)[:10]

    def _retrieve_relevant_documents_chroma(self, question: str, top_k: int = 5) -> List[Dict]:
        """Fast document retrieval with GPU acceleration"""
        if not self.chroma_collection:
            return []
            
        try:
            results = self.chroma_collection.query(
                query_texts=[question],
                n_results=top_k,
                include=["documents", "distances", "metadatas"]
            )
            
            relevant_docs = []
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                document = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                
                similarity = 1 / (1 + distance) if distance > 0 else 1.0
                
                relevant_docs.append({
                    "id": metadata["doc_id"],
                    "text": document[:1000] + "..." if len(document) > 1000 else document,
                    "score": float(similarity)
                })
            
            relevant_docs.sort(key=lambda x: x['score'], reverse=True)
            return relevant_docs
        except Exception as e:
            logger.error(f"ChromaDB query failed: {str(e)}")
            return []

    def _generate_structured_answer(self, question: str, graph_context: List[str], relevant_docs: List[Dict]) -> str:
        """Generate answer with GPU acceleration"""
        context_str = "## KNOWLEDGE GRAPH CONTEXT:\n"
        context_str += "\n".join(graph_context) if graph_context else "No relevant graph context found."
        
        context_str += "\n\n## RELEVANT DOCUMENT EXCERPTS:\n"
        if relevant_docs:
            for doc in relevant_docs:
                context_str += f"\n- Document: {doc['id']} (Relevance: {doc['score']:.2f})\n"
                context_str += f"  Excerpt: {doc['text']}\n"
        else:
            context_str += "No relevant documents found."
        
        prompt = (
            f"You are an academic assistant for UIET, Panjab University. "
            f"Answer the following question based on the provided context.\n\n"
            f"### QUESTION:\n{question}\n\n"
            f"### CONTEXT:\n{context_str}\n\n"
            f"### INSTRUCTIONS:\n"
            f"- Provide a clear, comprehensive answer\n"
            f"- Include supporting evidence from the context\n"
            f"- Format your response using Markdown\n"
            f"- Specify source documents when possible\n\n"
            f"### RESPONSE:"
        )
        
        try:
            response = requests.post(
                OLLAMA_ENDPOINT,
                json={
                    "model": ANSWER_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": 8192,
                        "temperature": 0.3,
                        "num_gpu": 50 if GPU_ENABLED else 0
                    }
                },
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
        
        return "I couldn't generate an answer for that question at this time."

# --------------------------
# Flask API Setup
# --------------------------
app = Flask(__name__)
rag_system = GraphRAGSystem()

# Add CORS headers to all responses
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# Explicit OPTIONS handler for /chat
@app.route('/chat', methods=['OPTIONS'])
def handle_chat_options():
    return '', 200

@app.route('/chat', methods=['POST'])
def handle_chat():
    """Handle chat requests"""
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Missing prompt parameter"}), 400
    
    if not rag_system.initialized:
        return jsonify({"error": "System not initialized"}), 400
    
    question = data['prompt']
    answer = rag_system.query_with_rag(question)
    
    return jsonify({
        "question": question,
        "answer": answer
    })

@app.route('/initialize', methods=['GET'])
def initialize_system():
    """Initialize the RAG system"""
    try:
        rag_system.initialize()
        return jsonify({
            "status": "success",
            "message": "System initialized",
            "gpu_enabled": GPU_ENABLED
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/graph', methods=['GET'])
def get_graph():
    """Get graph information"""
    if not rag_system.initialized:
        return jsonify({"error": "System not initialized"}), 400
    
    return jsonify({
        "graph_image": f"/static/{GRAPH_IMAGE}",
        "graph_data": f"/static/{GRAPH_FILE}",
        "stats": rag_system.get_graph_stats()
    })

@app.route('/status', methods=['GET'])
def system_status():
    """Get system status"""
    return jsonify({
        "initialized": rag_system.initialized,
        "gpu_enabled": GPU_ENABLED,
        "documents_processed": len(rag_system.documents),
        "graph_nodes": len(rag_system.knowledge_graph.nodes) if rag_system.initialized else 0,
        "graph_edges": len(rag_system.knowledge_graph.edges) if rag_system.initialized else 0
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory("static", filename)

# --------------------------
# Main Execution
# --------------------------
if __name__ == '__main__':
    # Start initialization in background
    def init_background():
        logger.info("Background initialization started...")
        rag_system.initialize()
        logger.info("Background initialization completed")
    
    threading.Thread(target=init_background, daemon=True).start()
    
    # Run Flask app
    logger.info(f"Starting API server at http://{API_HOST}:{API_PORT}")
    app.run(host=API_HOST, port=API_PORT, threaded=True)