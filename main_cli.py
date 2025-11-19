import sys
from main import process_image   # تابع پردازش تصویر شما

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: image-tool-linux <input> <output>")
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2]

    process_image(inp, out)
    print("Image processed successfully.")
