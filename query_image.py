import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone
pc = Pinecone(api_key="1619da1d-19d3-476b-8552-4468119bf7d6")

index_name = "image-search-index"

# Check if the index already exists; if not, create it

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=512,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)

# Move model to device if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


def create_image_embeddings(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs).cpu().numpy()
    return image_embedding.flatten().tolist()


def create_text_embeddings(text):
    inputs = processor(
        text=[text], return_tensors="pt",
        padding=True).to(device)
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs).cpu().numpy()
    return text_embedding.flatten().tolist()


query = "Show me a photo of a city"

# Generate image embedding
image_embedding = create_image_embeddings(
    "./image.jpg")

print(image_embedding)

# Query the Pinecone index
results = index.query(
    vector=image_embedding,
    top_k=3,
    include_values=False,
    include_metadata=True
)

print(results)
