# Zero Shot Natural Language Video Frame Retrieval System Using SigLIP

This project is a high-efficiency, zero-shot video frame retrieval system that enables natural language search across arbitrary video content using **SigLIP** (`ViT-B-16-SigLIP`). By projecting visual frames and free-form textual queries into a shared cross-modal embedding space, the system retrieves precise moments and timestamps without requiring task-specific fine-tuning or domain supervision.

### Methodology
To achieve real-time query speeds and lightweight storage footprints across long-form videos, the indexing pipeline employs a two-tier frame filtering architecture:
1. **Pixel-Difference Motion Filtering**: Low-level temporal frame differences filter out completely static background segments prior to deep neural network ingestion.
2. **Semantic Similarity Pruning**: Successive extracted frame embeddings undergo cosine similarity thresholding (pruning redundant frames with semantic similarity $\ge 0.88$), eliminating redundant representations while retaining key event transitions.
3. **Cross-Modal Retrieval**: Normalised SigLIP vision-language embeddings enable sub-second dot-product / cosine similarity ranking over indexed frames for any arbitrary textual query.

### Benchmark Results
Evaluated on the standardized **QVHighlights** benchmark across diverse real-world video domains, the system demonstrates strong zero-shot retrieval accuracy alongside notable efficiency:
- **Zero-Shot Frame Retrieval**: Achieves **72.00% Hit@1**, **93.00% Hit@5**, and **99.00% Hit@10**, with a Mean Reciprocal Rank (**MRR**) of **0.8171** and **0.7125 mAP**.
- **Indexing Efficiency**: Prunes an average of ~66% of video frames (~22.7 static frames and ~75.3 semantic duplicates pruned per video), cutting embedding compute and storage while maintaining an average indexing time of ~20.8 seconds per video.