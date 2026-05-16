import sys

def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "collect"

    if command == "collect":
        from src.collection.app import run
        run()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python main.py [collect]")
        sys.exit(1)

if __name__ == "__main__":
    main()
