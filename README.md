## DOCKER (MANUAL)

### Bring container UP
docker compose up -d

### On change, rebuild
docker compose up -d --build

### Bring container DOWN
docker compose down

### Bring container DOWN and wipe volume cache
docker compose down -v


## Smoke test
curl -X POST "http://localhost:8000/v1/chat" \
  -H "Authorization: Bearer super-secret-frontend-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "What is machine learning?"
      }
    ]
  }' \
  -N


curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Authorization: Bearer super-secret-frontend-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "embeddinggemma:300m",
    "input": "The quick brown fox jumps over the lazy dog"
  }' \
  -N


    Explain what additional contextual information you have been provided, show it to the user in this format "Context: {context}".
    Explain what the user question is asking in this format: "Question: {content}".