import os
import pandas as pd
import requests

# Test the pandas movie dataset download with the fixed URL
def test_pandas_download():
    url = "https://github.com/pandas-dev/pandas/raw/main/doc/data/movies.csv"
    file_path = "test_movies.csv"
    
    print(f"Testing download from: {url}")
    
    try:
        # Test the URL connectivity
        response = requests.head(url, allow_redirects=True, timeout=10)
        print(f"URL status code: {response.status_code}")
        
        if response.status_code == 200:
            print("URL is accessible!")
            
            # Try to download and read the file
            df = pd.read_csv(url)
            print(f"Successfully downloaded and read the file!")
            print(f"Data shape: {df.shape}")
            print(f"First few rows:")
            print(df.head())
            
            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed test file: {file_path}")
                
            return True
        else:
            print(f"URL returned error: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error during download test: {e}")
        return False

if __name__ == "__main__":
    success = test_pandas_download()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
