# from IPython.display import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="1619da1d-19d3-476b-8552-4468119bf7d6")

index_name = "image-search-index"

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


# Embed data
# We'll use an example dataset of images of animals and cities:


data = load_dataset(
    "jamescalam/image-text-demo",
    split="train"
)


model_id = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)

# move model to device if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


# ClIP allows for both text and image embeddings

def create_text_embeddings(text):
    text_embedding = processor(
        text=text,
        padding=True,
        images=None,
        return_tensors='pt').to(device)

    text_emb = model.get_text_features(**text_embedding)
    return text_emb[0]


def create_image_embeddings(image):
    vals = processor(
        text=None,
        images=image,
        return_tensors='pt')['pixel_values'].to(device)
    image_embedding = model.get_image_features(vals)
    return image_embedding[0]


# We will embed the images and search with text


def apply_vectorization(data):

    data["image_embeddings"] = create_image_embeddings(data["image"])
    return data


data = data.map(apply_vectorization)
# add an id column for easy indexing later
ids = [str(i) for i in range(0, data.num_rows)]
data = data.add_column("id", ids)


vectors = []
for i in range(0, data.num_rows):
    d = data[i]
    vectors.append({
        "id": d["id"],
        "values": d["image_embeddings"],
        "metadata": {"caption": d["text"]}
    })

index.upsert(
    vectors=vectors,
    namespace="ns1"
)
