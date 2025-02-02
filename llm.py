import json
import requests

def generate_response(prompt):
    url = "http://localhost:11434/api/generate"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3.1",  # Specify the model you want to use
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        response_data = response.json()
        return response_data['response']
    else:
        print("Error:", response.status_code, response.text)
        return None

# Predefined input
predefined_input = "Provide financial advice for someone who spends excessively on food and entertainment. Suggest ways to optimize their spending, save money, and build an emergency fund."

# Generate response and save to a text file
if __name__ == "__main__":
    response = generate_response(predefined_input)
    if response:
        print("Ollama: ", response)
        with open("output.txt", "w") as file:  # Open a file to write output
            file.write(response)  # Save only the response to the file
    else:
        print("No response generated.")


