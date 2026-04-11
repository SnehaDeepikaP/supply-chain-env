import uvicorn

def main():
    uvicorn.run(
        "app:app",   # points to your main FastAPI app
        host="0.0.0.0",
        port=7860
    )

if __name__ == "__main__":
    main()
