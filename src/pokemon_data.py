import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import os
import json
import requests
from pathlib import Path
from tqdm import tqdm
import time
import warnings
from io import BytesIO

# Suppress the specific PIL warning about palette images
warnings.filterwarnings('ignore', category=UserWarning, message='Palette images.*')

# Pokemon types
POKEMON_TYPES = [
    'normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison',
    'ground', 'flying', 'psychic', 'bug', 'rock', 'ghost', 'dragon', 'dark',
    'steel', 'fairy'
]

TYPE_TO_IDX = {t: i for i, t in enumerate(POKEMON_TYPES)}
NUM_TYPES = len(POKEMON_TYPES)

# Define sprite keys including official artwork
SPRITE_KEYS = ['front_default', 'front_shiny']
OFFICIAL_ART_KEY = 'other.official-artwork.front_default'

class PokemonDataset(Dataset):
    def __init__(self, root_dir="data/pokemon", image_size=64, download=True):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        
        # Enhanced data augmentation
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
        self.images_cache = {}  # Cache for transformed images
        
        # Verify all images are valid
        for pokemon in self.pokemon_data:
            primary_type = pokemon['type'][0].lower()
            type_idx = TYPE_TO_IDX[primary_type]
            
            # Check both regular sprites and official artwork
            for sprite_key in pokemon['sprites']:
                img_path = self.root_dir / "processed_images" / f"{pokemon['id']}_{sprite_key}.png"
                if img_path.exists():
                    try:
                        # Verify image is valid
                        with Image.open(img_path) as img:
                            # Convert to RGB during verification
                            img = img.convert('RGB')
                            # Save as RGB to avoid future conversions
                            img.save(img_path)
                        self.image_files.append(img_path)
                        self.type_labels.append(type_idx)
                    except (UnidentifiedImageError, OSError) as e:
                        print(f"Removing corrupted image {img_path}: {str(e)}")
                        os.remove(img_path)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            with Image.open(self.image_files[idx]) as image:
                # Image should already be RGB from initialization
                image = self.transform(image)
                type_label = self.type_labels[idx]
                
                return {
                    'image': image,
                    'type': torch.tensor(type_label, dtype=torch.long)
                }
        except (UnidentifiedImageError, OSError) as e:
            print(f"Error loading image {self.image_files[idx]}: {str(e)}")
            # Return a different image as fallback
            return self.__getitem__((idx + 1) % len(self))

    def _download_dataset(self):
        """Download all Pokemon sprites and metadata if not present"""
        if self.root_dir.exists():
            return
            
        print("Downloading Pokemon dataset...")
        self.root_dir.mkdir(parents=True, exist_ok=True)
        (self.root_dir / "images").mkdir(exist_ok=True)
        
        # First, get total number of Pokemon
        response = requests.get("https://pokeapi.co/api/v2/pokemon-species/")
        if response.status_code == 200:
            total_pokemon = response.json()['count']
        else:
            raise Exception("Failed to get Pokemon count")
            
        print(f"Found {total_pokemon} Pokemon to download")
        pokemon_data = []
        
        # Download all Pokemon with progress bar
        for i in tqdm(range(1, total_pokemon + 1), desc="Downloading Pokemon"):
            try:
                response = requests.get(f"https://pokeapi.co/api/v2/pokemon/{i}")
                if response.status_code == 200:
                    data = response.json()
                    sprites_data = {}
                    
                    # Get all available sprite variations
                    for sprite_key in SPRITE_KEYS:
                        sprite_url = data['sprites'].get(sprite_key)
                        if sprite_url:
                            sprites_data[sprite_key] = sprite_url
                    
                    # Get official artwork if available
                    official_art_url = data['sprites'].get('other', {}).get('official-artwork', {}).get('front_default')
                    if official_art_url:
                        sprites_data[OFFICIAL_ART_KEY] = official_art_url
                    
                    if sprites_data:  # Only add if we have at least one sprite
                        pokemon_info = {
                            'id': data['id'],
                            'name': data['name'],
                            'type': [t['type']['name'] for t in data['types']],
                            'sprites': sprites_data
                        }
                        
                        # Download all available sprites
                        for sprite_key, sprite_url in sprites_data.items():
                            sprite_response = requests.get(sprite_url)
                            if sprite_response.status_code == 200:
                                # Create filename based on sprite key
                                if sprite_key == OFFICIAL_ART_KEY:
                                    filename = f"{data['id']}_official_art.png"
                                else:
                                    filename = f"{data['id']}_{sprite_key}.png"
                                    
                                img_path = self.root_dir / "images" / filename
                                # Save directly as RGB
                                img_data = Image.open(BytesIO(sprite_response.content)).convert('RGB')
                                img_data.save(img_path)
                                
                                # Verify the saved image
                                try:
                                    with Image.open(img_path) as img:
                                        img.verify()
                                except (UnidentifiedImageError, OSError):
                                    print(f"Downloaded corrupted image for Pokemon {i}, sprite {sprite_key}, skipping")
                                    if img_path.exists():
                                        os.remove(img_path)
                                    continue
                        
                        pokemon_data.append(pokemon_info)
                    
                    # Add a small delay to avoid rate limiting
                    time.sleep(0.5)
                else:
                    print(f"Failed to download Pokemon {i}")
            except Exception as e:
                print(f"Error downloading Pokemon {i}: {str(e)}")
                continue
            
        # Save Pokemon data
        with open(self.root_dir / "pokemon.json", "w") as f:
            json.dump(pokemon_data, f)
            
def get_pokemon_dataloader(batch_size=64, image_size=64, num_workers=4, download=True):
    dataset = PokemonDataset(image_size=image_size, download=download)
    print(f"Dataset loaded with {len(dataset)} images")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    ) 

if __name__ == "__main__":
    # Download the dataset
    dataset = PokemonDataset(download=True)
    print(f"Dataset loaded with {len(dataset)} images")
