from src.config import Config
from src.extract_all import extract_all

def main():
    cfg = Config()
    extract_all(cfg)

if __name__ == "__main__":
    main()
