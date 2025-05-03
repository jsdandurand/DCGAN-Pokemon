import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import requests
from pathlib import Path

# Pokemon types
POKEMON_TYPES = [
    'normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison',
    'ground', 'flying', 'psychic', 'bug', 'rock', 'ghost', 'dragon', 'dark',
    'steel', 'fairy'
]

TYPE_TO_IDX = {t: i for i, t in enumerate(POKEMON_TYPES)}
NUM_TYPES = len(POKEMON_TYPES)

class PokemonDataset(Dataset):
    def __init__(self, root_dir="data/pokemon", image_size=64, download=True):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        if download:
            self._download_dataset()
            
        # Load Pokemon data
        with open(self.root_dir / "pokemon.json", "r") as f:
            self.pokemon_data = json.load(f)
            
        self.image_files = []
        self.type_labels = []
        
        for pokemon in self.pokemon_data:
            img_path = self.root_dir / "images" / f"{pokemon['id']}.png"
            if img_path.exists():
                self.image_files.append(img_path)
                # Get primary type
                primary_type = pokemon['type'][0].lower()
                type_idx = TYPE_TO_IDX[primary_type]
                self.type_labels.append(type_idx)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('RGB')
        image = self.transform(image)
        type_label = self.type_labels[idx]
        
        return {
            'image': image,
            'type': torch.tensor(type_label, dtype=torch.long)
        }

    def _download_dataset(self):
        """Download Pokemon sprites and metadata if not present"""
        if self.root_dir.exists():
            return
            
        print("Downloading Pokemon dataset...")
        self.root_dir.mkdir(parents=True, exist_ok=True)
        (self.root_dir / "images").mkdir(exist_ok=True)
        
        # Download Pokemon data
        pokemon_api = "https://pokeapi.co/api/v2/pokemon"
        pokemon_data = []
        
        # Get first 151 Pokemon (can be modified to get more)
        for i in range(1, 152):
            response = requests.get(f"{pokemon_api}/{i}")
            if response.status_code == 200:
                data = response.json()
                pokemon_info = {
                    'id': data['id'],
                    'name': data['name'],
                    'type': [t['type']['name'] for t in data['types']]
                }
                pokemon_data.append(pokemon_info)
                
                # Download sprite
                sprite_url = data['sprites']['front_default']
                if sprite_url:
                    sprite_response = requests.get(sprite_url)
                    if sprite_response.status_code == 200:
                        with open(self.root_dir / "images" / f"{data['id']}.png", "wb") as f:
                            f.write(sprite_response.content)
            
        # Save Pokemon data
        with open(self.root_dir / "pokemon.json", "w") as f:
            json.dump(pokemon_data, f)
            
def get_pokemon_dataloader(batch_size=64, image_size=64, num_workers=2):
    dataset = PokemonDataset(image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    ) 