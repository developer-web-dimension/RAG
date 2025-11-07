from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://4ada0d7c-dab1-42f8-b163-c9d568cb52ad.us-west-2-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.03L0dK2fEpX9ktL9hNx-0wDyQuXm04udOjpfyJu2xzs",
)

print(qdrant_client.get_collections())